"""Standalone ``pi mcp`` commands (F-CA-07): sign in, sign out, and list MCP
servers without entering the interactive REPL.

``pi mcp login <server> [--timeout seconds]`` limits the whole sign-in flow
(TS ``pi mcp login --timeout``); ``logout`` removes the stored grant; ``list``
prints the configured servers. Login resolves project trust like the CLI does
and only reads the project ``mcp.json`` when the folder is trusted.
"""

from __future__ import annotations

import asyncio
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


def mcp_list(cwd: str, agent_dir: str) -> int:
    config = Configuration(cwd, agent_dir)
    paths = [config.agent_dir / "mcp.json"]
    if config.project_trusted:
        paths.append(config.cwd / ".pi/mcp.json")
    for path in paths:
        entries = read_json(path).get("mcpServers", {})
        for name, value in entries.items():
            transport = "http" if "url" in value else "stdio"
            print(f"{name}: {transport}")
    return 0


def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: pi mcp login <server> [--timeout seconds] | logout <server> | list")
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
            return mcp_list(os.getcwd(), "")
        raise ValueError(f"unknown command: pi mcp {command}. Use login, logout, or list")
    except (ValueError, OSError, RuntimeError, AbortError, asyncio.TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
