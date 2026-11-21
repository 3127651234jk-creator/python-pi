"""Standalone ``pi mcp`` commands (F-CA-07): sign in, sign out, and list MCP
servers without entering the interactive REPL.

``pi mcp login <server> [--timeout seconds]`` limits the whole sign-in flow
(TS ``pi mcp login --timeout``); ``logout`` removes the stored grant; ``list``
prints the configured servers. Login resolves project trust like the CLI does
and only reads the project ``mcp.json`` when the folder is trusted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List, Optional

from ..util import AbortError
from .config import Configuration, read_json
from .mcp import McpManager
from .trust import TrustStore, resolve_project_trusted


def _parse_login_args(args: List[str]) -> "tuple[str, Optional[float]]":
    server: Optional[str] = None
    timeout: Optional[float] = None
    index = 0
    while index < len(args):
        if args[index] == "--timeout":
            if index + 1 >= len(args):
                raise ValueError("--timeout requires a value in seconds")
            try:
                timeout = float(args[index + 1])
            except ValueError:
                raise ValueError(f"--timeout expects a number, got {args[index + 1]}") from None
            if timeout <= 0:
                raise ValueError("--timeout must be positive")
            index += 2
        elif server is None:
            server = args[index]
            index += 1
        else:
            raise ValueError(f"unexpected argument: {args[index]}")
    if not server:
        raise ValueError("usage: pi mcp login <server> [--timeout seconds]")
    return server, timeout


def _load_server_settings(config: Configuration, server: str) -> dict:
    paths = [config.agent_dir / "mcp.json"]
    if config.project_trusted:
        paths.append(config.cwd / ".pi/mcp.json")
    for path in paths:
        entries = read_json(path).get("mcpServers", {})
        if isinstance(entries, dict):
            value = entries.get(server)
            if isinstance(value, dict):
                return value
    raise ValueError(f"Unknown MCP server: {server}")


async def mcp_login(server: str, timeout: Optional[float], cwd: str, agent_dir: str) -> int:
    config = Configuration(cwd, agent_dir)
    trusted = await resolve_project_trusted(
        cwd, TrustStore(config.agent_dir), None, config.settings, has_ui=False,
        notify=lambda message: print(message))
    config.project_trusted = trusted
    settings = _load_server_settings(config, server)
    if "url" not in settings:
        raise ValueError("only HTTP MCP servers support OAuth sign-in")
    manager = McpManager(config)
    manager.notify = print
    try:
        manager.login_servers[server] = settings
        await manager.oauth_login(server, timeout=timeout)
        print(f"credentials saved for MCP server {server}")
        return 0
    finally:
        await manager.close()


async def mcp_logout(server: str, cwd: str, agent_dir: str) -> int:
    config = Configuration(cwd, agent_dir)
    settings = _load_server_settings(config, server)
    if "url" not in settings:
        raise ValueError("only HTTP MCP servers support OAuth sign-in")
    manager = McpManager(config)
    try:
        manager.login_servers[server] = settings
        await manager.oauth_logout(server)
        print(f"credentials removed for MCP server {server}")
        return 0
    finally:
        await manager.close()


def _describe_transport(settings: dict) -> str:
    if "url" in settings:
        return settings["url"]
    if "command" not in settings:
        return ""
    return " ".join([settings["command"], *settings.get("args", [])]).strip()


def _configured_servers(config: Configuration) -> "dict[str, tuple[str, Optional[str], dict]]":
    """name -> (scope, override path, merged settings) from the config files.

    A project entry that only carries override keys refines a global server
    instead of defining its own (TS override/scope reporting). Settings are
    merged the way McpManager.reload does, so the listing can describe a server
    that never connected.
    """
    configured: "dict[str, tuple[str, Optional[str], dict]]" = {}
    global_entries = read_json(config.agent_dir / "mcp.json").get("mcpServers", {})
    if isinstance(global_entries, dict):
        for name, value in global_entries.items():
            if isinstance(value, dict):
                configured[name] = ("global", None, dict(value))
    if config.project_trusted:
        project_path = str(config.cwd / ".pi/mcp.json")
        project_entries = read_json(config.cwd / ".pi/mcp.json").get("mcpServers", {})
        if isinstance(project_entries, dict):
            for name, value in project_entries.items():
                if not isinstance(value, dict):
                    continue
                base = configured.get(name)
                base_settings = base[2] if base else {}
                if "command" not in value and "url" not in value and "type" not in value:
                    if base is None:
                        # An override without a global server; reload reports it.
                        configured[name] = ("global", None, {**base_settings, **value})
                        continue
                    configured[name] = (base[0], project_path, {**base_settings, **value})
                else:
                    configured[name] = ("project", None, {**base_settings, **value})
    return configured


def _list_state(manager: McpManager, name: str) -> str:
    if manager.errors.get(name) == "disabled":
        return "disabled"
    if name in manager.clients:
        return "lazy" if name in manager.lazy_servers else "connected"
    if name in manager.lazy_servers:
        return "lazy"
    if name in manager.login_servers:
        return "needs sign-in"
    error = manager.errors.get(name)
    if error and "requires sign-in" in error:
        return "needs sign-in"
    return "failed"


def _server_reports(manager: McpManager, config: Configuration) -> List[dict]:
    """Per-server listing records (TS McpServerEntry): connection state, tools,
    exposure, scope, and the resources capability marker."""
    from .mcp import _tool_exposure

    reports = []
    for name, (scope, override, settings) in _configured_servers(config).items():
        client = manager.clients.get(name)
        state = _list_state(manager, name)
        tools = [tool["name"] if isinstance(tool, dict) else tool.name
                 for tool in client.tools] if client is not None else []
        exposure = settings.get("exposure", "direct")
        report = {
            "name": name,
            "transport": _describe_transport(settings),
            "exposure": exposure,
            "scope": scope,
            "enabled": settings.get("enabled", True) is not False
            and manager.errors.get(name) != "disabled",
            "state": state,
            "tools": tools,
        }
        if override:
            report["override"] = override
        # Per-tool overrides only when they differ from the server's exposure
        # (TS toolExposure), so wildcards resolve to the tools they touch.
        tool_exposure = {}
        for tool in tools:
            value = _tool_exposure(settings, tool)
            if value != exposure:
                tool_exposure[tool] = value
        if tool_exposure:
            report["toolExposure"] = tool_exposure
        if client is not None and client.resources_supported:
            # Resources are not listed at connect: the marker says the server
            # offers them without counting every one (#10526).
            report["resources"] = True
        if state == "failed" and manager.errors.get(name):
            report["error"] = manager.errors[name]
        reports.append(report)
    return reports


async def mcp_list(cwd: str, agent_dir: str, json_output: bool = False) -> int:
    config = Configuration(cwd, agent_dir)
    trusted = await resolve_project_trusted(
        cwd, TrustStore(config.agent_dir), None, config.settings, has_ui=False,
        notify=lambda message: print(message))
    config.project_trusted = trusted
    manager = McpManager(config)
    try:
        await manager.reload()
        reports = _server_reports(manager, config)
    finally:
        await manager.close()
    server_names = {report["name"] for report in reports}
    config_errors = [f"{path}: {message}" for path, message in manager.errors.items()
                     if path not in server_names]
    failed = any(report["state"] == "failed" for report in reports)
    if json_output:
        print(json.dumps({"servers": reports, "errors": config_errors}, indent=2))
        return 1 if failed else 0
    if not reports and not config_errors:
        print(f"No MCP servers configured. Add them to {config.agent_dir / 'mcp.json'} "
              f"or {config.cwd / '.pi/mcp.json'}.")
    for report in reports:
        state = report["state"]
        if state == "connected":
            count = len(report["tools"])
            state = f"connected, {count} tool{'' if count == 1 else 's'}"
        print(f"{report['name']}: {state} ({report['exposure']}, {report['scope']})")
        if report["transport"]:
            print(f"  {report['transport']}")
        if report.get("override"):
            print(f"  project override: {report['override']}")
        if report["state"] == "needs sign-in":
            print(f"  sign in with: pi mcp login {report['name']}")
        if report["tools"]:
            tools = [f"{tool} [{report['toolExposure'][tool]}]"
                     if report.get("toolExposure", {}).get(tool) else tool
                     for tool in report["tools"]]
            print(f"  tools: {', '.join(tools)}")
        if report.get("resources"):
            print("  offers resources")
        if report.get("error"):
            print(f"  {report['error']}")
    for error in config_errors:
        print(f"config error: {error}")
    return 1 if failed else 0


def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: pi mcp login <server> [--timeout seconds] | logout <server> | list [--json]")
        return 0
    command, args = argv[0], argv[1:]
    try:
        if command == "login":
            server, timeout = _parse_login_args(args)
            return asyncio.run(mcp_login(server, timeout, os.getcwd(), ""))
        if command == "logout":
            if not args:
                raise ValueError("usage: pi mcp logout <server>")
            return asyncio.run(mcp_logout(args[0], os.getcwd(), ""))
        if command == "list":
            json_output = "--json" in args
            return asyncio.run(mcp_list(os.getcwd(), "", json_output=json_output))
        raise ValueError(f"unknown command: pi mcp {command}. Use login, logout, or list")
    except (ValueError, OSError, RuntimeError, AbortError, asyncio.TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
