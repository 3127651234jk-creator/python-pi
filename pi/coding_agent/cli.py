"""pi-python CLI: one-shot prompts and an interactive REPL.

Python port of the coding-agent CLI surface, simplified to a readline loop:
streaming text output, tool activity lines, slash commands, Ctrl+C abort.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import json
import getpass
from typing import Any, List, Optional
import httpx

from ..ai.models import ModelError, resolve_model
from ..ai.types import AssistantMessage, SystemMessage, TextContent, TextDeltaEvent, get_message_text
from ..agent.agent import Agent
from ..agent.compaction import compact_agent
from ..agent.types import (
    AgentEndEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from .catalog_sync import fetch_catalog, save_cache
from .codemode import create_codemode_tool
from .markdown_render import render_markdown
from .packages import PackageManager
from .session import SessionManager
from .system_prompt import SystemPromptOptions, build_system_prompt
from .tools import create_default_tools, tool_prompt_contributions
from .config import Configuration
from .model_registry import ModelRegistry
from .auth import login_device, login_openrouter
from .subscription_auth import get_or_create_device_id, login_claude, login_codex, login_copilot, login_kimi_coding, login_openai_chatgpt, login_radius, login_xai
from ..agent.compaction import CompactionSettings
from .resources import ResourceCatalog
from .mcp import McpManager
from .rpc import JsonWriter, RpcServer
from .extensions import ExtensionManager
from .tui import ExtensionAction, TerminalUI, THEMES, DEFAULT_KEYBINDINGS, load_themes
from .usage import context_usage, format_usage, session_stats, usage_cost_breakdown

BANNER = "pi-python — Python port of the pi coding agent core. Type /help for commands."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi", description="pi-python coding agent")
    parser.add_argument("prompt", nargs="*", help="Prompt to run once (omit for interactive mode)")
    parser.add_argument("--provider", help="anthropic | openai (OpenAI-compatible) | faux")
    parser.add_argument("--model", help="Model id, e.g. gpt-4o-mini or claude-sonnet-4-5-20250929")
    parser.add_argument("--base-url", help="Override the provider API base URL")
    parser.add_argument("--api-key", help="API key (defaults to provider environment variables)")
    parser.add_argument("--thinking", choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true", help="Continue latest session")
    parser.add_argument("--no-session", action="store_true", help="Do not persist this conversation")
    parser.add_argument("--session", help="Resume a session path or ID")
    parser.add_argument("--session-dir", help="Session storage directory")
    parser.add_argument("--name", help="Session name")
    parser.add_argument("--agent-dir", help="User configuration and credential directory")
    parser.add_argument("--login", metavar="PROVIDER", help="Store an API key or sign in with --oauth")
    parser.add_argument("--oauth", action="store_true", help="Use OAuth for --login")
    parser.add_argument("--logout", metavar="PROVIDER", help="Remove stored provider credentials")
    parser.add_argument("--list-models", action="store_true", help="List model catalog and capabilities")
    parser.add_argument("--mode", choices=["text", "json", "rpc"], default="text")
    parser.add_argument("--tui", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--sync-models", action="store_true", help="Refresh the cached online model catalog")
    parser.add_argument("--evals", metavar="FILE", help="Run eval cases from a JSON file and exit")
    return parser


def make_agent(args: argparse.Namespace, cwd: str, registry: Optional[ModelRegistry] = None,
               resources: Optional[ResourceCatalog] = None) -> Agent:
    registry = registry or ModelRegistry(Configuration(cwd, args.agent_dir or ""))
    settings = registry.config.settings
    provider = args.provider or os.environ.get("PI_PY_PROVIDER") or settings.get("provider", "openai")
    model_id = args.model or os.environ.get("PI_PY_MODEL") or (settings.get("model") if provider == settings.get("provider", provider) else None)
    base_url = args.base_url or os.environ.get("PI_PY_BASE_URL") or os.environ.get(provider.upper() + "_BASE_URL")
    model = registry.resolve(provider, model_id, args.api_key, base_url)
    tools = create_default_tools(cwd)
    snippets, guidelines = tool_prompt_contributions(tools)
    options = SystemPromptOptions(cwd=cwd, selected_tools=[tool.name for tool in tools], tool_snippets=snippets, tool_guidelines=guidelines)
    system_prompt = build_system_prompt(options)
    agent = Agent(model=model, tools=tools, system_prompt=system_prompt,
                 thinking_level=args.thinking or os.environ.get("PI_PY_THINKING") or settings.get("thinking", "off"),
                 compaction=CompactionSettings(**settings.get("compaction", {})))
    resources = resources or ResourceCatalog(registry.config)
    agent.messages[0].sections = {"skills": resources.system_prompt() or None}
    return agent


class CliRenderer:
    """Prints agent events to the terminal (markdown mode + tool output folding)."""

    def __init__(self, quiet: bool = False, settings: Optional[dict] = None) -> None:
        self.quiet = quiet
        self.settings = settings or {}
        self.markdown = bool(self.settings.get("markdown", False))
        self.fold = int(self.settings.get("fold_output", 10))

    async def __call__(self, event: Any) -> None:
        if self.quiet:
            return
        if isinstance(event, MessageUpdateEvent):
            inner = event.assistant_message_event
            if isinstance(inner, TextDeltaEvent) and not self.markdown:
                print(inner.delta, end="", flush=True)
        elif isinstance(event, MessageEndEvent) and self.markdown and isinstance(event.message, AssistantMessage):
            text = "\n".join(block.text for block in event.message.content if isinstance(block, TextContent))
            if text.strip():
                print("\n" + render_markdown(text), flush=True)
        elif isinstance(event, ToolExecutionStartEvent):
            args = event.args or {}
            shown = args.get("command") or args.get("path") or ""
            print(f"\n[{event.tool_name}] {shown}", flush=True)
        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            text = ""
            if result is not None and getattr(result, "content", None):
                text = result.content[0].text
            if event.is_error:
                print(f"  error: {text}", flush=True)
            elif self.fold and result is not None and text:
                lines = text.splitlines()
                for line in lines[: self.fold]:
                    print(f"  {line}", flush=True)
                if len(lines) > self.fold:
                    print(f"  [+{len(lines) - self.fold} lines folded; /set fold_output 0 shows all]", flush=True)
        elif isinstance(event, TurnEndEvent):
            message = event.message
            usage = getattr(message, "usage", None)
            if usage is not None and (usage.input or usage.output):
                cost = usage.cost.get("total", 0.0) if isinstance(usage.cost, dict) else 0.0
                print(
                    f"\n[{usage.total_tokens} tokens, in {usage.input}, out {usage.output}, ${cost:.4f}]",
                    flush=True,
                )
        elif isinstance(event, AgentEndEvent):
            print()


async def run_prompt(agent: Agent, text: str, renderer: CliRenderer, session: Optional[SessionManager],
                     resources: Optional[ResourceCatalog] = None, extensions: Optional[ExtensionManager] = None,
                     mcp: Optional[McpManager] = None) -> None:
    """Run one prompt with abort handling and session persistence."""

    if mcp is not None and mcp.lazy_servers:
        for name in list(mcp.lazy_servers):
            try:
                await mcp.ensure_tools(name)
            except (OSError, ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                print(f"MCP {name}: {exc}", file=sys.stderr)
    if extensions:
        text = await extensions.process_input(text)
        if text is None:
            return
    if resources:
        text = resources.expand(text)
    unsubscribe = agent.subscribe(_session_recorder(session))
    unsubscribe_render = agent.subscribe(renderer)
    task = asyncio.ensure_future(agent.prompt(text))
    try:
        await task
    except KeyboardInterrupt:
        agent.abort()
        try:
            await task
        except Exception:
            pass
        print("\n[aborted]", flush=True)
    finally:
        unsubscribe()
        unsubscribe_render()


def _session_recorder(session: Optional[SessionManager]):

    async def record(event: Any) -> None:
        if session is not None and isinstance(event, MessageEndEvent):
            try:
                session.append(event.message)
            except (OSError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Could not save session: {exc}") from exc

    return record


SLASH_HELP = """Commands:
  /help          show this help
  /model <id>    switch model id (provider unchanged)
  /models        list catalog and capabilities
  /thinking [level]  show or set the thinking level
  /login <provider> [oauth]  store API key or sign in
  /logout <provider>        remove saved credentials
  /settings [key JSON]     view or save user defaults
  /reload        reload settings and model catalog
  /skills        list available skills
  /prompts       list prompt template commands
  /mcp [reconnect|login <server>|logout <server>]  MCP status and management
  /package [install <source>|update <name>|remove <name>|list]  manage packages
  /extensions    list loaded extensions and commands
  /theme [dark|light|mono|<user>]  choose terminal theme
  /keys          show configurable terminal shortcuts
  /skill:<name> [request]   load a skill explicitly
  /compact [instructions]  summarize older context now
  /usage         show cumulative token usage and cost
  /sync-models   refresh the cached online model catalog
  /evals <file>  run eval cases against this agent
  /clear         start a fresh conversation
  /new           start a fresh session
  /resume [id]   list or switch saved sessions
  /session       show active session details
  /name <name>   name the current session
  /tree [id]     show tree or continue after an entry
  /fork [id]     copy a branch to a new session
  /clone         copy the active branch to a new session
  /export <path> export active branch (.jsonl, .html, .md)
  /export-ts <path>  export the active branch in TS pi session format
  /exit, /quit   leave interactive mode
Anything else is sent as a prompt. Ctrl+C aborts the current run."""


async def interactive_loop(agent: Agent, renderer: CliRenderer, session: Optional[SessionManager], registry: Optional[ModelRegistry] = None,
                           resources: Optional[ResourceCatalog] = None, mcp: Optional[McpManager] = None,
                           extensions: Optional[ExtensionManager] = None, ui: Optional[TerminalUI] = None,
                           packages: Optional[PackageManager] = None) -> None:
    loop = asyncio.get_running_loop()
    print(BANNER)
    while True:
        try:
            value = await ui.read() if ui else await loop.run_in_executor(None, lambda: input("\n> ").strip())
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if isinstance(value, ExtensionAction):
            try:
                await value.handler(agent, extensions)
            except Exception as exc:  # noqa: BLE001 - extension shortcuts report their own errors
                print(f"error: {exc}")
            continue
        text = value or ""
        if not text:
            continue
        if text.startswith("/"):
            command, _, rest = text.partition(" ")
            if command == "/keys":
                print(json.dumps({**DEFAULT_KEYBINDINGS, **(registry.config.settings.get("keybindings", {}) if registry else {})}, indent=2))
                continue
            if command == "/theme":
                available = ui.themes if ui else THEMES
                selected = rest.strip() or (await ui.choose("Theme", [(name, name) for name in available]) if ui else None)
                if selected in available:
                    if registry:
                        registry.config.save("theme", selected)
                    if ui:
                        ui.set_theme(selected)
                continue
            if ui and not rest.strip():
                if command == "/model" and registry:
                    registry.reload()
                    selected = await ui.choose("Model", [(name, f"{name} ({model.context_window} tokens)") for name, model in sorted(registry.models.items())])
                    if not selected:
                        continue
                    rest = selected
                elif command == "/resume" and session:
                    selected = await ui.choose("Session", [(item["path"], f"{item['name'] or item['id']} ({item['messages']} messages)") for item in session.list_sessions()])
                    if not selected:
                        continue
                    rest = selected
                elif command == "/tree" and session:
                    selected = await ui.choose("Conversation tree", [(entry.id, f"{entry.id[:8]} {entry.message.role}: {get_message_text(entry.message.content)[:70]}") for entry in session.entries if entry.message is not None])
                    if not selected:
                        continue
                    rest = selected
            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                print(SLASH_HELP)
                if extensions:
                    for name, (_, description) in extensions.commands.items():
                        print(f"  /{name}  {description}")
                if resources:
                    for template in resources.prompts.values():
                        print(f"  /{template.name} {template.argument_hint}  {template.description}")
                    if resources.config.settings.get("enable_skill_commands", True):
                        for skill in resources.skills.values():
                            print(f"  /skill:{skill.name}  {skill.description}")
                continue
            if command == "/extensions":
                if extensions:
                    for api in extensions.extensions:
                        print(api.path)
                continue
            if command == "/mcp":
                if mcp:
                    parts = rest.split()
                    try:
                        if parts and parts[0] == "reconnect":
                            await mcp.reload()
                            sync_mcp_tools(agent, mcp)
                        elif parts and parts[0] == "login" and len(parts) > 1:
                            await mcp.oauth_login(parts[1])
                            print(f"credentials saved for MCP server {parts[1]}")
                        elif parts and parts[0] == "logout" and len(parts) > 1:
                            mcp.oauth_logout(parts[1])
                            print(f"credentials removed for MCP server {parts[1]}")
                        for name, client in mcp.clients.items():
                            state = "lazy" if name in mcp.lazy_servers else f"connected, {len(client.tools)} tools"
                            print(f"{name}: {state}")
                        for name, error in mcp.errors.items():
                            print(f"{name}: {error}")
                    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                        print(f"error: {exc}")
                continue
            if command in ("/skills", "/prompts"):
                if resources:
                    items = resources.skills.values() if command == "/skills" else resources.prompts.values()
                    for item in items:
                        print(f"{item.name}: {item.description}\n  {item.path}")
                continue
            if command == "/model":
                if rest.strip():
                    try:
                        if registry:
                            registry.reload()
                            selected = rest.strip()
                            provider, model_id = selected.split("/", 1) if selected.split("/", 1)[0] in registry.providers and "/" in selected else (agent.model.provider, selected)
                            agent.model = registry.resolve(provider, model_id)
                        else:
                            agent.model = resolve_model(agent.model.provider, rest.strip(), agent.model.api_key, agent.model.base_url)
                    except (ValueError, ModelError) as exc:
                        print(f"error: {exc}")
                        continue
                    print(f"model: {agent.model.id}")
                    if session:
                        session.append_model_change(agent.model.provider, agent.model.id)
                else:
                    print(f"model: {agent.model.id}")
                continue
            if command == "/thinking":
                level = rest.strip() or agent.thinking_level
                if level not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
                    print("error: thinking level must be off|minimal|low|medium|high|xhigh|max")
                    continue
                agent.thinking_level = level
                if session:
                    session.append_thinking_change(level)
                print(f"thinking: {level}")
                continue
            if command == "/package":
                try:
                    await package_command(rest, packages or PackageManager(registry.config if registry else Configuration(os.getcwd())), resources)
                except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command == "/compact":
                if agent.is_streaming:
                    print("error: agent is streaming")
                    continue
                try:
                    result = await compact_agent(agent, session, rest.strip())
                    print(f"compacted: ~{result['tokens_before']} -> ~{result['tokens_after']} estimated tokens")
                    if result["summary"]:
                        preview = result["summary"] if len(result["summary"]) <= 2000 else result["summary"][:2000] + "..."
                        print("\n" + preview)
                except (RuntimeError, ValueError, ModelError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command == "/usage":
                totals = usage_cost_breakdown(agent.messages, registry)
                print(format_usage(totals, context_usage(agent.messages, agent.model)))
                continue
            if command == "/sync-models":
                if registry is None:
                    print("error: model registry unavailable")
                    continue
                try:
                    providers = await fetch_catalog()
                    save_cache(registry.config.agent_dir, providers)
                    registry.reload()
                    total = sum(len(entry["models"]) for entry in providers.values())
                    print(f"synced {total} models across {len(providers)} providers")
                except (OSError, ValueError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command == "/evals":
                if not rest.strip():
                    print("error: usage: /evals <file>")
                    continue
                try:
                    passed = await run_evals_file(agent, renderer, session, rest.strip(), mcp)
                    print(f"evals {'passed' if passed else 'failed'}")
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f"error: {exc}")
                continue
            if command in ("/models", "/login", "/logout", "/settings", "/reload"):
                try:
                    if registry is None:
                        raise ValueError("Model registry unavailable")
                    if command == "/models":
                        registry.reload()
                        print_models(registry)
                    elif command == "/login":
                        parts = rest.split()
                        if not parts:
                            raise ValueError("Usage: /login <provider> [oauth]")
                        await authenticate(registry, parts[0], "oauth" in parts[1:])
                    elif command == "/logout":
                        if not rest.strip():
                            raise ValueError("Usage: /logout <provider>")
                        registry.auth.logout(rest.strip())
                    elif command == "/settings":
                        key, _, value = rest.strip().partition(" ")
                        if key:
                            registry.config.save(key, json.loads(value))
                        print(json.dumps(registry.config.settings, ensure_ascii=False, indent=2))
                    else:
                        registry.config.reload()
                        registry.reload()
                        agent.model = registry.resolve(agent.model.provider, agent.model.id)
                        agent.thinking_level = registry.config.settings.get("thinking", agent.thinking_level)
                        agent.compactor.settings = CompactionSettings(**registry.config.settings.get("compaction", {}))
                        if resources:
                            resources.reload()
                            for diagnostic in resources.diagnostics:
                                print(f"warning: {diagnostic}", file=sys.stderr)
                            update = SystemMessage(content="", timestamp=0, sections={"skills": resources.system_prompt() or None})
                            agent.messages.append(update)
                            if session:
                                session.append(update)
                        print("configuration reloaded")
                        if mcp:
                            await mcp.reload()
                            sync_mcp_tools(agent, mcp)
                        if extensions:
                            await extensions.reload()
                            registry.register_extension_providers(extensions.all_providers())
                except (OSError, ValueError, ModelError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command in ("/clear", "/new"):
                agent.messages = [m for m in agent.messages if getattr(m, "role", None) == "system"]
                if session is not None:
                    session.start_new()
                    for message in agent.messages:
                        session.append(message)
                print("conversation cleared")
                continue
            if command == "/export-ts":
                if session is None:
                    print("error: session persistence is disabled")
                    continue
                if not rest.strip():
                    print("error: usage: /export-ts <path.jsonl>")
                    continue
                try:
                    print(session.export_ts(rest.strip()))
                except (OSError, ValueError) as exc:
                    print(f"error: {exc}")
                continue
            if command in ("/resume", "/session", "/name", "/tree", "/fork", "/clone", "/export"):
                try:
                    handle_session_command(command, rest.strip(), agent, session)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"error: {exc}")
                continue
            if not (extensions and command[1:] in extensions.commands) and (not resources or (not command.startswith("/skill:") and command[1:] not in resources.prompts)):
                print(f"unknown command: {command}")
                continue
        try:
            prompt = run_prompt(agent, text, renderer, session, resources, extensions, mcp)
            if ui:
                await ui.run(prompt)
            else:
                await prompt
        except (OSError, RuntimeError, ModelError, ValueError, httpx.HTTPError) as exc:
            print(f"error: {exc}")


async def package_command(argument: str, config: Configuration, resources: Optional[ResourceCatalog]) -> None:
    """Handle /package: list | install <source> | update <name> | remove <name>."""
    manager = PackageManager(config)
    parts = argument.split()
    subcommand = parts[0] if parts else "list"
    if subcommand == "list":
        installed = manager.list()
        if not installed:
            print("no packages installed")
            return
        for package in installed:
            kinds = ", ".join(sorted(package.files)) or "no resources"
            print(f"{package.name}  {package.source}  {package.version[:12]}  [{kinds}]")
            for kind, paths in sorted(package.files.items()):
                for path in paths:
                    print(f"  {kind}: {path}")
    elif subcommand == "install":
        if len(parts) < 2:
            raise ValueError("Usage: /package install <npm:pkg | git-url>")
        package = await manager.install(parts[1])
        print(f"installed {package.name} ({package.version[:12]})")
        _refresh_resources(resources, package)
    elif subcommand == "update":
        if len(parts) < 2:
            raise ValueError("Usage: /package update <name>")
        package = await manager.update(parts[1])
        print(f"updated {package.name} ({package.version[:12]})")
        _refresh_resources(resources, package)
    elif subcommand == "remove":
        if len(parts) < 2:
            raise ValueError("Usage: /package remove <name>")
        manager.remove(parts[1])
        print(f"removed {parts[1]}")
        if resources:
            resources.reload()
    else:
        raise ValueError("Usage: /package [list|install|update|remove]")


def _refresh_resources(resources: Optional[ResourceCatalog], package) -> None:
    if resources is not None and (package.files.get("skills") or package.files.get("prompts")):
        resources.reload()
        for diagnostic in resources.diagnostics:
            print(f"warning: {diagnostic}", file=sys.stderr)


async def run_evals_file(agent: Agent, renderer: CliRenderer, session: Optional[SessionManager],
                         path: str, mcp: Optional[McpManager] = None) -> bool:
    """Run eval cases from a JSON file against the current agent."""
    from ..evals import load_cases, run_evals, summarize

    cases = load_cases(path)

    async def runner(prompt: str):
        before = len(agent.messages)
        await run_prompt(agent, prompt, renderer, session, mcp=mcp)
        return agent.messages[before:]

    results = await run_evals(cases, runner)
    print(summarize(results))
    return all(result.passed for result in results)


def print_models(registry: ModelRegistry) -> None:
    for identifier, model in sorted(registry.models.items()):
        print(f"{identifier}  context={model.context_window} output={model.max_tokens} reasoning={model.reasoning} input={','.join(model.input)}")


async def authenticate(registry: ModelRegistry, provider: str, oauth: bool = False) -> None:
    provider = {"claude": "anthropic", "codex": "openai-codex", "copilot": "github-copilot",
        "chatgpt": "openai", "openai-chatgpt": "openai"}.get(provider, provider)
    if provider not in registry.providers:
        raise ModelError(f"Unknown provider: {provider}")
    if oauth or provider in ("openai-codex", "github-copilot", "kimi-coding", "xai", "radius"):
        async def prompt(message):
            return await asyncio.get_running_loop().run_in_executor(None, getpass.getpass, message)
        if provider == "anthropic":
            credential = await login_claude(print, prompt)
        elif provider == "openai":
            credential = await login_openai_chatgpt(print, prompt, get_or_create_device_id(registry.config.agent_dir))
        elif provider == "openai-codex":
            credential = await login_codex(print)
        elif provider == "github-copilot":
            credential = await login_copilot(print)
        elif provider == "kimi-coding":
            credential = await login_kimi_coding(print)
        elif provider == "xai":
            credential = await login_xai(print)
        elif provider == "radius":
            async def choose(message):
                return await asyncio.get_running_loop().run_in_executor(None, input, message)
            credential = await login_radius(print, choose)
        elif provider == "openrouter":
            credential = await login_openrouter(print)
        elif registry.providers[provider].get("oauth"):
            credential = await login_device(registry.providers[provider]["oauth"], print)
        else:
            raise ModelError(f"OAuth not configured for {provider}; API-key login is available")
    else:
        key = await asyncio.get_running_loop().run_in_executor(None, getpass.getpass, f"API key for {provider}: ")
        if not key.strip():
            raise ValueError("Empty API key")
        credential = {"type": "api_key", "key": key.strip()}
    registry.auth.set(provider, credential)
    print(f"credentials saved for {provider}")


def handle_session_command(command: str, argument: str, agent: Agent, session: Optional[SessionManager]) -> None:
    if session is None:
        raise ValueError("Session persistence is disabled")
    if command == "/resume":
        if argument:
            agent.messages = session.resume(argument)
        else:
            for item in session.list_sessions():
                print(f"{item['id']}  {item['name']}  {item['messages']} messages")
    elif command == "/session":
        stats = session_stats(agent.messages, agent.model, session.session_file)
        print(f"{session.session_file}\nname: {session.name}\nleaf: {session.leaf_id}\nmessages: {stats['total_messages']}")
        print(f"user: {stats['user_messages']}  assistant: {stats['assistant_messages']}  tool calls: {stats['tool_calls']}")
        print(f"tokens: in {stats['tokens']['input']:,}  out {stats['tokens']['output']:,}  "
              f"cache {stats['tokens']['cache_read']:,}/{stats['tokens']['cache_write']:,}  cost: ${stats['cost']:.4f}")
    elif command == "/name":
        if not argument:
            raise ValueError("Usage: /name <name>")
        session.rename(argument)
    elif command == "/tree":
        if argument:
            agent.messages = session.branch(argument)
        else:
            print(session.tree())
    elif command in ("/fork", "/clone"):
        agent.messages = session.fork(argument or None if command == "/fork" else None)
        print(f"session: {session.session_file}")
    elif command == "/export":
        if not argument:
            raise ValueError("Usage: /export <path.jsonl|path.html|path.md>")
        print(session.export(argument))
    if command in ("/resume", "/tree", "/fork", "/clone") and (argument or command in ("/fork", "/clone")):
        agent.compactor.prefix = []
        agent.compactor.summary = ""


def restore_session_settings(agent: Agent, session: SessionManager, registry: ModelRegistry) -> None:
    """Replay model_change / thinking_level_change / compaction entries on resume."""
    model_change = session.latest_custom("model_change")
    if model_change and model_change.get("provider"):
        try:
            agent.model = registry.resolve(model_change["provider"], model_change.get("modelId") or None)
        except (ModelError, ValueError) as exc:
            print(f"warning: could not restore model from session: {exc}", file=sys.stderr)
    thinking = session.latest_custom("thinking_level_change")
    if thinking and thinking.get("thinkingLevel") in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        agent.thinking_level = thinking["thinkingLevel"]
    compaction = session.latest_custom("compaction")
    if compaction and compaction.get("summary"):
        agent.compactor.summary = compaction["summary"]


def sync_mcp_tools(agent: Agent, mcp: McpManager) -> None:
    """Replace the agent's MCP tools with the manager's current set."""
    for tool in mcp.tools:
        tool._mcp_managed = True
    agent.tools[:] = [tool for tool in agent.tools if not getattr(tool, "_mcp_managed", False)] + list(mcp.tools)


async def async_main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_session and (args.session or args.continue_session or args.name):
        print("error: --no-session cannot be combined with session options", file=sys.stderr)
        return 2
    cwd = os.getcwd()
    if args.mode == "rpc" and args.prompt or args.mode == "json" and not args.prompt:
        print("error: rpc accepts prompts through stdin; json requires a prompt argument", file=sys.stderr)
        return 2
    if args.mode != "text" and (args.login or args.logout or args.list_models or args.sync_models or args.evals):
        print("error: authentication, model-list, sync and eval commands require text mode", file=sys.stderr)
        return 2

    try:
        registry = ModelRegistry(Configuration(cwd, args.agent_dir or ""))
        if args.sync_models:
            providers = await fetch_catalog()
            save_cache(registry.config.agent_dir, providers)
            registry.reload()
            total = sum(len(entry["models"]) for entry in providers.values())
            print(f"synced {total} models across {len(providers)} providers")
            return 0
        if args.login:
            await authenticate(registry, args.login, args.oauth)
            return 0
        if args.logout:
            registry.auth.logout(args.logout)
            return 0
        if args.list_models:
            print_models(registry)
            return 0
        if args.oauth:
            raise ModelError("--oauth requires --login")
        resources = ResourceCatalog(registry.config)
        for diagnostic in resources.diagnostics:
            print(f"warning: {diagnostic}", file=sys.stderr)
        agent = make_agent(args, cwd, registry, resources)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session: Optional[SessionManager] = None
    if not args.no_session:
        session = SessionManager(session_dir=args.session_dir or registry.config.settings.get("session_dir"), cwd=cwd)
        if args.session:
            agent.messages = session.resume(args.session)
        elif args.continue_session:
            latest = session.latest_session_file()
            if latest is None:
                print("no previous session found", file=sys.stderr)
                session.start_new()
                for message in agent.messages:
                    session.append(message)
            else:
                agent.messages = session.load(latest)
                print(f"continued session: {os.path.basename(latest)}", file=sys.stderr)
        else:
            session.start_new()
            for message in agent.messages:
                session.append(message)
        if args.name:
            session.rename(args.name)
        if args.session or args.continue_session:
            restore_session_settings(agent, session, registry)
            update = SystemMessage(content="", timestamp=0, sections={"skills": resources.system_prompt() or None})
            agent.messages.append(update)
            session.append(update)

        def persist_summary(summary: str) -> None:
            session.append_compaction(summary)

        agent.compactor.on_summary = persist_summary

    renderer = CliRenderer(settings=registry.config.settings)

    mcp = McpManager(registry.config)
    extensions = ExtensionManager(registry.config, agent, session)
    try:
        await mcp.reload()
        sync_mcp_tools(agent, mcp)
        mcp.on_tools_changed = lambda server: sync_mcp_tools(agent, mcp)
        await extensions.reload()
        registry.register_extension_providers(extensions.all_providers())
        if registry.config.settings.get("codemode", {}).get("enabled"):
            agent.tools.append(create_codemode_tool(
                agent.tools, session, registry.config.settings.get("codemode", {})))
        for name, error in mcp.errors.items():
            print(f"MCP {name}: {error}", file=sys.stderr)
        if args.evals:
            passed = await run_evals_file(agent, renderer, session, args.evals, mcp)
            return 0 if passed else 1
        if args.mode == "rpc":
            server = RpcServer(agent, JsonWriter(), session, registry, resources, extensions)
            extensions.set_ui_sink(server.ui_sink)
            await server.serve()
            return 0
        if args.mode == "json":
            writer = JsonWriter()
            try:
                await run_prompt(agent, " ".join(args.prompt), writer, session, resources, extensions, mcp)
            except Exception as exc:
                await writer.write({"type": "run_error", "error": str(exc)})
                return 1
            finally:
                await writer.write({"type": "agent_settled"})
            return 0
        if args.prompt:
            await run_prompt(agent, " ".join(args.prompt), renderer, session, resources, extensions, mcp)
            return 0
        ui = None
        if args.tui == "on" or args.tui == "auto" and sys.stdin.isatty() and sys.stdout.isatty():
            def commands():
                names = [line.strip().split()[0] for line in SLASH_HELP.splitlines() if line.strip().startswith("/")]
                names.extend("/" + name for name in extensions.commands)
                names.extend("/" + name for name in resources.prompts)
                if registry.config.settings.get("enable_skill_commands", True):
                    names.extend("/skill:" + name for name in resources.skills)
                return names
            ui = TerminalUI(agent, commands, registry.config.settings,
                            extra_bindings=extensions.all_keybindings(),
                            themes=load_themes(registry.config.agent_dir))
        await interactive_loop(agent, renderer, session, registry, resources, mcp, extensions, ui,
                               PackageManager(registry.config))
        return 0
    finally:
        await extensions.close()
        await mcp.close()


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        print()
        return 130
    except (OSError, ValueError, ModelError, asyncio.TimeoutError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
