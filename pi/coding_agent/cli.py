"""pi-python CLI: one-shot prompts and an interactive REPL.

Python port of the coding-agent CLI surface, simplified to a readline loop:
streaming text output, tool activity lines, slash commands, Ctrl+C abort.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import shutil
import sys
import json
import getpass
import threading
from contextlib import nullcontext, redirect_stdout
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import httpx
from prompt_toolkit import PromptSession

from ..ai.models import ModelError, resolve_image_model, resolve_model
from ..ai.providers import generate_images, get_default_stream_fn
from ..ai.retry import RetryPolicy
from ..ai.types import AssistantMessage, ImageContent, ImagesContext, SystemMessage, TextContent, TextDeltaEvent, get_message_text
from ..agent.agent import Agent
from ..agent.compaction import compact_agent, prefix_checkpoint, restore_compaction
from ..util import AbortError
from ..agent.types import (
    AgentEndEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from .catalog_sync import fetch_catalog, save_cache
from .radius_catalog import refresh_radius_catalog
from ..codemode import create_codemode_tool
from .nested_tool_calls import execute_tool as execute_nested_tool
from .crash_log import clear_crash_log, crash_log_path, read_crash_log, record_crash, take_unnotified_crash
from .diagnostics import Diagnostic, deduplicate_diagnostics, report_diagnostics, settings_diagnostics
from .bug_report import (BUG_REPORT_CUSTOM_ENTRY_TYPE, bug_report_archive_file_name,
                         collect_bug_report_diagnostics, collect_bug_report_metadata,
                         generate_bug_report_summary, upload_bug_report, write_bug_report_archive)
from .file_processor import process_file_arguments
from .markdown_render import render_markdown
from .syntax_highlight import MONO_THEME
from .terminal_image import render_image_lines
from .output_guard import raw_stdout, restore_stdout, take_over_stdout
from .packages import PackageManager
from .provider_attribution import apply_attribution, is_opencode_like
from .session import SESSION_ID_PATTERN, SessionManager, export_session_file
from .system_prompt import SystemPromptOptions, build_system_prompt
from .tools import BUILTIN_TOOL_NAMES, create_default_tools, tool_prompt_contributions
from .trust import TrustStore, project_trust_parent_path, resolve_project_trusted, trust_options
from .config import Configuration, read_json
from .custom_api import (RESERVED_PROVIDER_ID, ConcurrentConfigChange, CustomApiSaver,
                         CustomApiForm, SaveError, WizardIO, run_custom_api_wizard,
                         validate_form_values)
from .custom_api_form import SaveResult
from .model_registry import ModelRegistry
from .auth import login_device, login_openrouter
from .subscription_auth import get_or_create_device_id, login_claude, login_codex, login_copilot, login_kimi_coding, login_openai_chatgpt, login_radius, login_xai
from ..agent.compaction import CompactionSettings
from . import timings
from .resources import ResourceCatalog
from .mcp import McpManager, create_tool_name_matcher, is_mcp_tool_name
from .tool_search import STORE_TYPE as TOOL_SEARCH_STORE_TYPE, create_tool_search_tool
from .bash_mode import execute_bang as run_bang, render_bang_output
from .program_status import ProgramStatusReporter, program_status_enabled
from .rpc import JsonWriter, RpcServer, WireJsonWriter
from .extensions import ExtensionManager
from .ts_extension import to_jsonable as to_json_value
from .tui import ExtensionAction, TerminalUI, THEMES, DEFAULT_KEYBINDINGS, load_themes
from .usage import context_usage, format_usage, session_stats, summary_usage_extra, usage_cost_breakdown

BANNER = "pi-python — Python port of the pi coding agent core. Type /help for commands."

# Crash-report context for main(): filled in as async_main creates the registry
# and session, so an uncaught exception can be persisted with its location.
CRASH_CONTEXT: dict = {}


def _version_string() -> str:
    try:
        return package_version("pi-python")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi", description="pi-python coding agent")
    parser.add_argument("prompt", nargs="*", help="Prompt to run once (omit for interactive mode)")
    parser.add_argument("--provider", help="Provider id: openai, anthropic, openrouter, google, google-vertex, "
                                           "openai-responses, azure-openai, azure-openai-responses, amazon-bedrock, "
                                           "openai-codex, github-copilot, kimi-coding, xai, radius, faux or models.json entry")
    parser.add_argument("--model", help="Model id, e.g. gpt-4o-mini or claude-sonnet-4-5-20250929")
    parser.add_argument("--base-url", help="Override the provider API base URL")
    parser.add_argument("--api-key", help="API key (defaults to provider environment variables)")
    parser.add_argument("--thinking", choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true", help="Continue latest session")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Pick a previous session to continue from a list")
    parser.add_argument("--no-session", action="store_true", help="Do not persist this conversation")
    parser.add_argument("--session", help="Resume a session path or ID")
    parser.add_argument("--session-id", help="Continue the session with this exact id, or create a new one with it")
    parser.add_argument("--fork", metavar="SESSION",
                        help="Copy a session path or id into a new session in this directory and continue it")
    parser.add_argument("--session-dir", help="Session storage directory")
    parser.add_argument("--name", help="Session name")
    parser.add_argument("--agent-dir", help="User configuration and credential directory")
    parser.add_argument("--login", metavar="PROVIDER", help="Store an API key or sign in with --oauth")
    parser.add_argument("--oauth", action="store_true", help="Use OAuth for --login")
    parser.add_argument("--logout", metavar="PROVIDER", help="Remove stored provider credentials")
    parser.add_argument("--list-models", action="store_true", help="List model catalog and capabilities")
    parser.add_argument("--mode", choices=["text", "json", "rpc"], default="text")
    parser.add_argument("--tui", choices=["auto", "on", "off", "fullscreen"], default="auto")
    parser.add_argument("--approve", "-a", dest="approve", action="store_true", default=None,
                        help="Trust the project folder for this run")
    parser.add_argument("--no-approve", "-na", dest="approve", action="store_false",
                        help="Do not trust the project folder for this run")
    parser.add_argument("--sync-models", action="store_true", help="Refresh the cached online model catalog")
    parser.add_argument("--evals", metavar="FILE", help="Run eval cases from a JSON file and exit")
    parser.add_argument("--no-tools", "-nt", action="store_true",
                        help="Disable all tools by default (built-in, extension and MCP)")
    parser.add_argument("--no-builtin-tools", "-nbt", action="store_true",
                        help="Disable the built-in tools (read, bash, edit, write, grep, find, ls, powershell)")
    parser.add_argument("--tools", "-t", metavar="TOOLS",
                        help="Comma-separated allowlist of tool names or patterns (*) to enable. "
                             "Keeps MCP tools unless an entry starts with mcp__. "
                             "Alternatively, a list of only +name/-name entries modifies the "
                             "built-in set (e.g. --tools +codemode,-find); it cannot mix with "
                             "plain names or patterns")
    parser.add_argument("--exclude-tools", "-xt", metavar="TOOLS",
                        help="Comma-separated denylist of tool names or patterns (*) to disable. "
                             "Applies to all tools, MCP tools included")
    parser.add_argument("--no-mcp", action="store_true",
                        help="Disable built-in MCP support: no servers connect and no MCP tools")
    parser.add_argument("-e", "--extension", action="append", metavar="PATH",
                        help="Load this extension path (repeatable)")
    parser.add_argument("--no-extensions", "-ne", action="store_true",
                        help="Skip discovered extensions; explicit --extension paths still load")
    parser.add_argument("--skill", action="append", metavar="PATH",
                        help="Load this skill file or directory (repeatable)")
    parser.add_argument("--no-skills", "-ns", action="store_true",
                        help="Skip discovered skills; --skill paths still load")
    parser.add_argument("--prompt-template", action="append", metavar="PATH",
                        help="Load this prompt template file or directory (repeatable)")
    parser.add_argument("--no-prompt-templates", "-np", action="store_true",
                        help="Skip discovered prompt templates; --prompt-template paths still load")
    parser.add_argument("--theme", action="append", metavar="PATH",
                        help="Load this theme file or directory (repeatable)")
    parser.add_argument("--use-theme", metavar="NAME", help="Use this theme for this run")
    parser.add_argument("--no-themes", action="store_true", help="Skip user theme loading")
    parser.add_argument("--no-context-files", "-nc", action="store_true",
                        help="Do not load AGENTS.md / CLAUDE.md / CONTEXT.md into the system prompt")
    parser.add_argument("--system-prompt", help="Replace the default system prompt (text or file path)")
    parser.add_argument("--append-system-prompt", action="append",
                        help="Append to the system prompt (text or file path; repeatable)")
    parser.add_argument("--models", metavar="PATTERNS",
                        help="Comma-separated provider/model patterns (*) the model cycle is limited to")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true",
                        help="Run one prompt non-interactively (stdin becomes the prompt when none is given)")
    parser.add_argument("--export", metavar="FILE",
                        help="Export a session file (HTML by default; .jsonl for TS format, .md) and exit; "
                             "the output path is the first positional argument")
    parser.add_argument("--verbose", action="store_true", help="Print startup timing groups")
    parser.add_argument("--offline", action="store_true", help="Run without network access")
    parser.add_argument("--version", "-v", action="version", version=_version_string(),
                        help="Show the version and exit")
    return parser


def split_extension_flags(argv: List[str], parser: argparse.ArgumentParser):
    """Keep TS-compatible unknown long flags out of argparse until extensions load."""
    known = {option for action in parser._actions for option in action.option_strings}
    filtered, values = [], {}
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            filtered.extend(argv[index:])
            break
        name, separator, inline_value = argument.partition("=")
        if argument.startswith("--") and name not in known:
            flag_name = name[2:]
            if not flag_name:
                filtered.append(argument)
            elif separator:
                values[flag_name] = inline_value
            elif index + 1 < len(argv) and not argv[index + 1].startswith(("-", "@")):
                values[flag_name] = argv[index + 1]
                index += 1
            else:
                values[flag_name] = True
        else:
            filtered.append(argument)
        index += 1
    return filtered, values


class ToolSelection:
    """The parsed ``--tools`` / ``--exclude-tools`` / ``--no-tools`` selection.

    ``--tools`` keeps MCP tools unless the allowlist is empty (``--no-tools``) or an entry
    starts with ``mcp__`` (only matching MCP tools are kept). ``--exclude-tools`` applies to
    everything, MCP tools included.

    MCP tools the allowlist keeps without naming them stay registered for codemode scripts
    but are not declared to the model, matching the TS session that keeps them out of the
    declared loadout. tool_search can load searchable tools for direct model calls.
    """

    def __init__(self, allowed: Optional[List[str]], excluded: List[str]) -> None:
        self.allowed = create_tool_name_matcher(allowed) if allowed is not None else None
        self.allowlist_filters_mcp = allowed is not None and (
            len(allowed) == 0 or any(entry.startswith("mcp__") for entry in allowed))
        self.excluded = create_tool_name_matcher(excluded) if excluded else None

    def keeps(self, name: str) -> bool:
        """Whether a tool stays registered under this selection."""
        if self.excluded is not None and self.excluded(name):
            return False
        if self.allowed is None or self.allowed(name):
            return True
        return not self.allowlist_filters_mcp and is_mcp_tool_name(name)

    def declares(self, name: str) -> bool:
        """Whether requests declare the tool to the model under this selection."""
        if self.excluded is not None and self.excluded(name):
            return False
        if self.allowed is None or self.allowed(name) or self.allowlist_filters_mcp:
            return True
        return not is_mcp_tool_name(name)


def split_tool_entries(value: str) -> List[str]:
    """Comma-separated ``--tools`` / ``--exclude-tools`` entries; blanks dropped."""
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def is_tool_modifier(entry: str) -> bool:
    """Whether a tool selection entry is a ``+name`` or ``-name`` modifier."""
    return entry.startswith("+") or entry.startswith("-")


def get_tool_list_error(entries: List[str]) -> Optional[str]:
    """Validate a ``--tools`` list: plain names/patterns, or only ``+name``/
    ``-name`` modifiers with exact names. Returns the problem, or None."""
    modifiers = [entry for entry in entries if is_tool_modifier(entry)]
    if not modifiers:
        return None
    if len(modifiers) < len(entries):
        return "tool names cannot be mixed with +name or -name entries"
    pattern = next((entry for entry in modifiers if "*" in entry), None)
    if pattern:
        return f"+name and -name entries take exact tool names, not patterns: {pattern}"
    return None


def apply_tool_modifiers(base: List[str], entries: List[str]) -> List[str]:
    """Apply ``+name``/``-name`` entries to ``base`` in order: ``+name`` adds a
    tool, ``-name`` removes one; other entries are ignored."""
    tools = list(base)
    for entry in entries:
        if not is_tool_modifier(entry):
            continue
        name = entry[1:]
        index = tools.index(name) if name in tools else -1
        if entry.startswith("+") and index == -1 and name:
            tools.append(name)
        elif entry.startswith("-") and index != -1:
            tools.pop(index)
    return tools


def tool_selection_from_args(args: argparse.Namespace) -> Optional[ToolSelection]:
    """The selection for one run, or None when no tool flag was given."""
    if args.tools is None and not args.no_tools and not args.exclude_tools and not getattr(args, "no_builtin_tools", False):
        return None
    allowed = split_tool_entries(args.tools) if args.tools is not None else ([] if args.no_tools else None)
    if allowed is not None:
        error = get_tool_list_error(allowed)
        if error:
            raise ValueError(error)
        if allowed and all(is_tool_modifier(entry) for entry in allowed):
            # Modifier lists are incremental: the built-in tools are the base
            # set, and each entry adds or removes one tool (TS applyToolModifiers).
            allowed = apply_tool_modifiers(list(BUILTIN_TOOL_NAMES), allowed)
    excluded = split_tool_entries(args.exclude_tools) if args.exclude_tools else []
    if getattr(args, "no_builtin_tools", False):
        # --no-builtin-tools drops exactly the built-in tools; extension and MCP
        # tools stay registered and declared.
        excluded = list(dict.fromkeys(excluded + list(BUILTIN_TOOL_NAMES)))
    return ToolSelection(allowed, excluded)


def resolve_prompt_input(value: str) -> str:
    """A --system-prompt / --append-system-prompt value: file path or literal text."""
    if os.path.isfile(value):
        try:
            with open(value, "r", encoding="utf-8-sig") as handle:
                return handle.read()
        except OSError as exc:
            raise ValueError(f"Could not read {value}: {exc}") from None
    return value


def apply_tool_selection(agent: Agent) -> None:
    """Re-apply the run's tool selection to the agent's current tool set."""
    selection = getattr(agent, "_tool_selection", None)
    if selection is not None:
        agent.tools[:] = [tool for tool in agent.tools if selection.keeps(tool.name)]
    session = getattr(agent, "_tool_search_session", None)
    loaded = (session.latest_custom(TOOL_SEARCH_STORE_TYPE) or {}).get("loaded", []) if session else getattr(agent, "_loaded_tools", [])
    # Kept MCP tools stay executable for codemode but are not declared to the
    # model when the allowlist did not name them.
    for tool in agent.tools:
        selected = tool.declared if selection is None else selection.declares(tool.name)
        if tool.exposure in ("deferred", "codemode"):
            tool.declared = tool.name in loaded
        else:
            tool.declared = selected
            if not selected and is_mcp_tool_name(tool.name):
                tool.exposure = "codemode"
                tool.declared = tool.name in loaded
    refresh_system_prompt(agent)


def skill_file_read_tool(agent: Agent) -> Optional[str]:
    """The tool that loads skill files.

    ``read``/``bash`` when declared; ``indirect`` when either is active but
    hidden (reachable through codemode); None when neither is available at all.
    """

    declared = {tool.name for tool in agent.tools if tool.declared}
    if "read" in declared:
        return "read"
    if "bash" in declared:
        return "bash"
    active = {tool.name for tool in agent.tools}
    if "read" in active or "bash" in active:
        return "indirect"
    return None


def refresh_system_prompt(agent: Agent) -> None:
    """Rebuild the tool list, rules and skills hint from the current loadout.

    Hidden tools are selected but not declared: they leave the tool list and
    rules while codemode still reaches them. Called after MCP refreshes,
    extension reloads and selection changes.
    """

    options = getattr(agent, "_system_prompt_options", None)
    if options is None:
        return
    tools = agent.tools
    for tool in tools:
        refresh = getattr(tool, "_codemode_refresh", None)
        if refresh is not None:
            refresh()
    snippets, guidelines = tool_prompt_contributions(tools)
    options.selected_tools = [tool.name for tool in tools]
    options.hidden_tools = [tool.name for tool in tools if not tool.declared]
    options.tool_snippets = snippets
    options.tool_guidelines = guidelines
    prompt = build_system_prompt(options)
    agent.system_prompt = prompt
    if agent.messages and getattr(agent.messages[0], "role", None) == "system":
        agent.messages[0].content = prompt
        resources = getattr(agent, "_resources", None)
        if resources is not None:
            agent.messages[0].sections = {
                **(agent.messages[0].sections or {}),
                "skills": resources.system_prompt(file_read_tool=skill_file_read_tool(agent)) or None,
            }


def make_agent(args: argparse.Namespace, cwd: str, registry: Optional[ModelRegistry] = None,
               resources: Optional[ResourceCatalog] = None) -> Agent:
    registry = registry or ModelRegistry(Configuration(cwd, args.agent_dir or ""))
    settings = registry.config.settings
    provider = args.provider or os.environ.get("PI_PY_PROVIDER") or settings.get("provider", "openai")
    model_id = args.model or os.environ.get("PI_PY_MODEL") or (settings.get("model") if provider == settings.get("provider", provider) else None)
    base_url = args.base_url or os.environ.get("PI_PY_BASE_URL") or os.environ.get(provider.upper() + "_BASE_URL")
    model = registry.resolve(provider, model_id, args.api_key, base_url)
    image_settings = settings.get("images", {})
    # The read tool consults these lazily so /model switches and /settings
    # changes take effect without recreating the tools.
    current_agent: dict = {}
    tools = create_default_tools(
        cwd,
        get_model=lambda: current_agent.get("agent"),
        image_settings=lambda: registry.config.settings.get("images", {}),
    )
    # The tool list and rules follow the selected tools, so the selection applies
    # before the system prompt is built. MCP and extension tools are filtered by
    # apply_tool_selection as they are added.
    selection = tool_selection_from_args(args)
    if selection is not None:
        tools = [tool for tool in tools if selection.keeps(tool.name)]
    snippets, guidelines = tool_prompt_contributions(tools)
    append_values = [resolve_prompt_input(value) for value in getattr(args, "append_system_prompt", None) or []]
    options = SystemPromptOptions(
        cwd=cwd, selected_tools=[tool.name for tool in tools],
        hidden_tools=[tool.name for tool in tools if not tool.declared],
        tool_snippets=snippets, tool_guidelines=guidelines,
        custom_prompt=resolve_prompt_input(args.system_prompt) if getattr(args, "system_prompt", None) else None,
        append_system_prompt="\n\n".join(append_values),
        context_files=[] if getattr(args, "no_context_files", False) else None)
    system_prompt = build_system_prompt(options)
    agent = Agent(model=model, tools=tools, system_prompt=system_prompt,
                 thinking_level=args.thinking or os.environ.get("PI_PY_THINKING") or settings.get("thinking", "off"),
                 compaction=CompactionSettings(**settings.get("compaction", {})),
                 cache_retention=settings.get("cache_retention", "5m") or None,
                 block_images=bool(image_settings.get("block_images", False)),
                 retry=RetryPolicy(**settings.get("retry", {})))
    apply_attribution(model, settings)
    # Kept on the agent so every later tool change re-applies the same selection
    # and rebuilds the prompt from the new loadout.
    agent._tool_selection = selection
    agent._system_prompt_options = options
    current_agent["agent"] = agent
    agent._loaded_tools = []
    def activate_tools(names: List[str]) -> None:
        loaded = list(dict.fromkeys([tool.name for tool in agent.tools
                                    if tool.declared and tool.exposure in ("deferred", "codemode")] + names))
        session = getattr(agent, "_tool_search_session", None)
        if session is not None:
            session.append_custom(TOOL_SEARCH_STORE_TYPE, {"loaded": loaded})
        agent._loaded_tools = loaded
        apply_tool_selection(agent)
    agent._activate_searched_tools = activate_tools
    if selection is not None and selection.allowed is not None and selection.keeps("tool_search"):
        agent.tools.append(create_tool_search_tool(agent.tools, activate_tools))
    resources = resources or ResourceCatalog(registry.config)
    agent._resources = resources
    agent.messages[0].sections = {"skills": resources.system_prompt(file_read_tool=skill_file_read_tool(agent)) or None}
    return agent


def format_duration(ms: float) -> str:
    """Adaptive duration label for the shell tools' Took line (port of the TS
    shell renderers' formatDuration)."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    remainder = total_seconds % 60
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    return f"{minutes // 60}h {minutes % 60}m {remainder}s"


class CliRenderer:
    """Prints agent events to the terminal (markdown mode + tool output folding)."""

    def __init__(self, quiet: bool = False, settings: Optional[dict] = None) -> None:
        self.quiet = quiet
        self.settings = settings or {}
        self.markdown = bool(self.settings.get("markdown", False))
        self.fold = int(self.settings.get("fold_output", 10))
        # Horizontal padding for transcript content (outputPad setting, default 1).
        self.output_pad = 0 if self.settings.get("output_pad") == 0 else 1
        self.extensions: Optional[ExtensionManager] = None
        self.tool_args = {}
        self.ui: Optional[TerminalUI] = None
        self.full_screen = False

    def pad(self, text: str) -> str:
        """Left padding for transcript content lines (outputPad); blank lines
        stay blank so block spacing is unaffected."""
        if self.output_pad <= 0 or not text:
            return text
        prefix = " " * self.output_pad
        return "\n".join(prefix + line if line else line for line in text.split("\n"))

    def image_width(self) -> int:
        if self.ui is not None:
            return self.ui.width()
        return shutil.get_terminal_size((80, 24)).columns

    def render_image_block(self, block: ImageContent) -> str:
        """Render an image content block as terminal lines.

        Kitty/iTerm2 placements need raw output, so the fullscreen transcript
        (painted by prompt_toolkit) uses truecolor half-block cells instead.
        """
        lines = render_image_lines(
            block.data, block.mime_type,
            max_width_cells=max(8, min(60, self.image_width() - 2)),
            allow_protocol=not self.full_screen,
        )
        return "\n".join(lines)

    def print_image_block(self, block: ImageContent) -> None:
        print(self.render_image_block(block), flush=True)

    async def __call__(self, event: Any) -> None:
        if self.quiet:
            return
        if isinstance(event, MessageUpdateEvent):
            inner = event.assistant_message_event
            transformed = self.extensions is not None and any(
                api.markdown_transformers for api in self.extensions.extensions)
            if isinstance(inner, TextDeltaEvent) and not self.markdown and not transformed:
                print(inner.delta, end="", flush=True)
        elif isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            if event.message.stop_reason == "error":
                # Provider failures arrive as a message with stop_reason "error";
                # without this line the REPL returns to the prompt silently.
                print(f"\nerror: {event.message.error_message or 'Unknown provider error'}", flush=True)
            transformed = self.extensions is not None and any(
                api.markdown_transformers for api in self.extensions.extensions)
            if self.markdown or transformed:
                text = "\n".join(block.text for block in event.message.content if isinstance(block, TextContent))
                if text.strip():
                    if self.extensions is not None:
                        text = await self.extensions.transform_markdown(text, "assistant")
                    theme = MONO_THEME if self.settings.get("theme") == "mono" else None
                    print("\n" + self.pad(render_markdown(text, theme=theme)), flush=True)
            for block in event.message.content:
                if isinstance(block, ImageContent):
                    self.print_image_block(block)
        elif isinstance(event, ToolExecutionStartEvent):
            args = event.args or {}
            self.tool_args[event.tool_call_id] = args
            indent = "  " if event.parent_tool_call_id else ""
            if self.extensions is not None:
                lines = await self.extensions.render_tool(event.tool_name, "call", args,
                    {"args": args, "toolCallId": event.tool_call_id, "expanded": not bool(self.fold),
                     "isPartial": False, "executionStarted": True, "argsComplete": True,
                     "parentToolCallId": event.parent_tool_call_id, "outputPad": self.output_pad}, 80)
                if lines is not None:
                    print("\n" + "\n".join(indent + line for line in lines), flush=True)
                    return
            shown = args.get("command") or args.get("path") or args.get("pattern") or ""
            print("\n" + self.pad(f"{indent}[{event.tool_name}] {shown}"), flush=True)
        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            indent = "  " if event.parent_tool_call_id else ""
            if self.extensions is not None and result is not None:
                options = {"args": self.tool_args.pop(event.tool_call_id, {}), "toolCallId": event.tool_call_id,
                           "expanded": not bool(self.fold), "isPartial": False, "isError": event.is_error,
                           "executionStarted": True, "argsComplete": True,
                           "parentToolCallId": event.parent_tool_call_id, "outputPad": self.output_pad}
                if event.duration_ms is not None:
                    options["durationMs"] = event.duration_ms
                lines = await self.extensions.render_tool(event.tool_name, "result", to_json_value(result),
                    options, 80)
                if lines is not None:
                    if lines:
                        print("\n".join(indent + line for line in lines), flush=True)
                    return
            text = ""
            if result is not None and getattr(result, "content", None):
                text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
            if event.is_error:
                print(self.pad(f"{indent}  error: {text}"), flush=True)
            elif self.fold and result is not None and text:
                lines = text.splitlines()
                for line in lines[: self.fold]:
                    print(self.pad(f"{indent}  {line}"), flush=True)
                if len(lines) > self.fold:
                    print(f"{indent}  [+{len(lines) - self.fold} lines folded; /set fold_output 0 shows all]", flush=True)
            if event.duration_ms is not None and event.tool_name in ("bash", "powershell"):
                # The recorded execute() duration: monotonic and reloaded from
                # the session record, unlike the wall-clock wait the command
                # itself reports (matching the TS shell renderers' Took line).
                print(f"{indent}  Took {format_duration(event.duration_ms)}", flush=True)
            for block in getattr(result, "content", None) or []:
                if isinstance(block, ImageContent):
                    self.print_image_block(block)
        elif isinstance(event, ToolExecutionUpdateEvent) and self.extensions is not None:
            lines = await self.extensions.render_tool(event.tool_name, "result",
                to_json_value(event.partial_result), {
                    "args": self.tool_args.get(event.tool_call_id, {}),
                    "toolCallId": event.tool_call_id, "expanded": not bool(self.fold),
                    "isPartial": True, "isError": False, "executionStarted": True,
                    "argsComplete": True, "parentToolCallId": event.parent_tool_call_id}, 80)
            if lines:
                indent = "  " if event.parent_tool_call_id else ""
                print("\n".join(indent + line for line in lines), flush=True)
        elif isinstance(event, AutoRetryStartEvent):
            print(
                f"\n[retrying attempt {event.attempt}/{event.max_attempts} in "
                f"{event.delay_ms / 1000:.1f}s: {event.error_message}]",
                flush=True,
            )
        elif isinstance(event, AutoRetryEndEvent) and not event.success:
            reason = "retry cancelled" if event.final_error == "Retry cancelled" else f"retry failed: {event.final_error}"
            print(f"\n[{reason}]", flush=True)
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
                     mcp: Optional[McpManager] = None,
                     images: Optional[List[Any]] = None) -> None:
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
    if images and isinstance(renderer, CliRenderer):
        for image in images:
            if isinstance(image, ImageContent):
                renderer.print_image_block(image)
    unsubscribe = agent.subscribe(_session_recorder(session))
    unsubscribe_render = agent.subscribe(renderer)
    task = asyncio.ensure_future(agent.prompt(text, images=images))
    try:
        # asyncio.run translates Ctrl+C into cancellation of the main task.
        # Keep the agent alive long enough to finish its cooperative abort and
        # publish the partial message through the subscribed session recorder.
        await asyncio.shield(task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        agent.abort()
        try:
            await task
        except (Exception, asyncio.CancelledError):
            pass
        if isinstance(renderer, JsonWriter):
            raise RuntimeError("Request aborted") from None
        print("\n[aborted]", flush=True)
    finally:
        unsubscribe()
        unsubscribe_render()
        if extensions is not None:
            # Extension-facing settle event (TS AgentSession emits it after
            # every run; the JSON wire event is written by the caller).
            await extensions.emit("agent_settled",
                                  {"type": "agent_settled", "aborted": agent.abort_requested})


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
  /login [provider] [oauth]  sign in (provider picker when omitted; /login custom
                 adds a custom API model with base URL, key and settings)
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
  /image [model] <prompt>  generate an image via openrouter-images
  /sync-models   refresh the cached online model catalog
  /evals <file>  run eval cases against this agent
  /trust         save a project trust decision for this folder
  /bug [hint]    report a bug to the pi developers (upload or zip)
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


async def read_console_input() -> str:
    loop = asyncio.get_running_loop()
    result = loop.create_future()

    def deliver(value, error):
        if result.done():
            return
        if error is not None:
            result.set_exception(error)
        else:
            result.set_result(value)

    def read():
        value, error = None, None
        try:
            value = input("\n> ").strip()
        except BaseException as exc:
            error = exc
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:
            pass  # The loop has already closed after cancellation.

    threading.Thread(target=read, daemon=True).start()
    return await result


async def interactive_loop(agent: Agent, renderer: CliRenderer, session: Optional[SessionManager], registry: Optional[ModelRegistry] = None,
                           resources: Optional[ResourceCatalog] = None, mcp: Optional[McpManager] = None,
                           extensions: Optional[ExtensionManager] = None, ui: Optional[TerminalUI] = None,
                           packages: Optional[PackageManager] = None,
                           initial: Optional[Tuple[str, List[Any]]] = None) -> None:
    loop = asyncio.get_running_loop()
    print(BANNER)
    reporter = None
    if program_status_enabled() and sys.stdout.isatty():
        def emit_status(sequence: str) -> None:
            try:
                sys.stdout.write(sequence)
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
        reporter = ProgramStatusReporter(emit_status, lambda: session.name if session else None)
        agent.subscribe(reporter.handle_event)
        if ui:
            ui.program_status = reporter
    if initial is not None and (initial[0] or initial[1]):
        try:
            await run_prompt(agent, initial[0], renderer, session, resources, extensions, mcp, images=initial[1])
        except (OSError, RuntimeError, ModelError, ValueError, httpx.HTTPError) as exc:
            print(f"error: {exc}")
    while True:
        try:
            value = await ui.read() if ui else await read_console_input()
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
        if text.lstrip().startswith("!"):
            # Shell escape (F-TUI-04): run outside the model; `!!` keeps the
            # output out of the context. The run lands in the transcript and
            # the session as a bashExecution message.
            try:
                message = await run_bang(text, os.getcwd())
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                print(f"error: {exc}")
                continue
            agent.messages.append(message)
            if session:
                session.append_bash_execution(message)
            print("\n" + renderer.pad(render_bang_output(message)), flush=True)
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
                    print(f"theme: {selected}")
                elif rest.strip():
                    print(f"error: unknown theme: {rest.strip()}. Available: {', '.join(available)}")
                elif not ui:
                    current = registry.config.settings.get("theme", "dark") if registry else "dark"
                    print(f"theme: {current}. Available: {', '.join(available)}")
                continue
            if ui and not rest.strip():
                if command == "/model" and registry:
                    registry.reload()
                    selected = await ui.choose("Model", [(name, f"{name} ({model.context_window} tokens)") for name, model in sorted(registry.scoped_model_items())])
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
                elif command == "/login" and registry:
                    method = await ui.choose("Sign in", LOGIN_METHODS)
                    if not method:
                        continue
                    if method == "custom":
                        rest = "custom"
                    else:
                        selected = await ui.choose("Provider", login_provider_options(registry, method == "oauth"))
                        if not selected:
                            continue
                        rest = selected + (" oauth" if method == "oauth" else "")
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
                            server, login_timeout = _mcp_login_args(parts[1:])
                            await mcp.oauth_login(server, timeout=login_timeout)
                            sync_mcp_tools(agent, mcp)
                            print(f"credentials saved for MCP server {server}")
                        elif parts and parts[0] == "logout" and len(parts) > 1:
                            await mcp.oauth_logout(parts[1])
                            sync_mcp_tools(agent, mcp)
                            print(f"credentials removed for MCP server {parts[1]}")
                        for name, state in mcp.server_states().items():
                            print(f"{name}: {state}")
                        for name, error in mcp.errors.items():
                            if name in mcp.clients:
                                print(f"{name}: {error}")
                    except AbortError:
                        print("error: MCP sign-in cancelled")
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
                totals = usage_cost_breakdown(agent.messages, registry,
                                              extra_usage=summary_usage_extra(session) if session else None)
                print(format_usage(totals, context_usage(agent.messages, agent.model)))
                continue
            if command == "/image":
                try:
                    await image_command(rest, registry, renderer)
                except (ModelError, OSError, ValueError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command == "/sync-models":
                if registry is None:
                    print("error: model registry unavailable")
                    continue
                try:
                    providers = await fetch_catalog()
                    save_cache(registry.config.agent_dir, providers)
                    radius_count = 0
                    try:
                        radius_count = await refresh_radius_catalog(registry)
                    except (OSError, ValueError, httpx.HTTPError) as exc:
                        print(f"warning: Radius model discovery failed: {describe_error(exc)}", file=sys.stderr)
                    registry.reload()
                    total = sum(len(entry["models"]) for entry in providers.values())
                    print(f"synced {total} models across {len(providers)} providers")
                    if radius_count:
                        print(f"synced {radius_count} Radius models from the gateway")
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
                        if not parts and not ui:
                            # Plain terminal: numbered menu instead of the picker.
                            # _choose_console takes (label, value) pairs.
                            method = await _choose_console("Sign in", [(label, value) for value, label in LOGIN_METHODS])
                            if method is None:
                                continue
                            if method == "custom":
                                parts = ["custom"]
                            else:
                                selected = await _choose_console("Provider", login_provider_options(registry, method == "oauth"))
                                if selected is None:
                                    continue
                                parts = [selected] + (["oauth"] if method == "oauth" else [])
                        if not parts:
                            raise ValueError("Usage: /login <provider> [oauth]\nProviders: "
                                             + ", ".join(sorted(registry.providers)))
                        if parts[0] == "custom":
                            await custom_login_command(registry, agent, session, ui)
                        else:
                            await authenticate(registry, parts[0], "oauth" in parts[1:])
                    elif command == "/logout":
                        if not rest.strip():
                            raise ValueError("Usage: /logout <provider>")
                        registry.auth.logout(rest.strip())
                    elif command == "/settings":
                        key, _, value = rest.strip().partition(" ")
                        if key:
                            registry.config.save(key, json.loads(value))
                            if key == "images":
                                sync_block_images(agent, registry)
                        print(json.dumps(registry.config.settings, ensure_ascii=False, indent=2))
                    else:
                        registry.config.reload()
                        registry.reload()
                        agent.model = registry.resolve(agent.model.provider, agent.model.id)
                        agent.thinking_level = registry.config.settings.get("thinking", agent.thinking_level)
                        agent.compactor.settings = CompactionSettings(**registry.config.settings.get("compaction", {}))
                        sync_block_images(agent, registry)
                        retry = RetryPolicy(**registry.config.settings.get("retry", {}))
                        agent.retry = retry
                        agent.compactor.retry = retry
                        if resources:
                            resources.reload()
                            for diagnostic in resources.diagnostics:
                                print(f"warning: {diagnostic}", file=sys.stderr)
                        print("configuration reloaded")
                        if mcp:
                            await mcp.reload(wait=False)
                            sync_mcp_tools(agent, mcp)
                        if extensions:
                            await extensions.reload()
                            if ui and ui.full_screen:
                                extensions.set_ui_sink(ui.ui_request)
                            registry.register_extension_providers(extensions.all_providers())
                        for tool in agent.tools:
                            update = getattr(tool, "_codemode_update_settings", None)
                            if update is not None:
                                update(registry.config.settings.get("codemode", {}))
                        apply_tool_selection(agent)
                        if resources:
                            # The skills hint follows the tools that are declared now.
                            update = SystemMessage(content="", timestamp=0,
                                                   sections={"skills": resources.system_prompt(
                                                       file_read_tool=skill_file_read_tool(agent)) or None})
                            agent.messages.append(update)
                            if session:
                                session.append(update)
                except (OSError, ValueError, ModelError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                    print(f"error: {exc}")
                continue
            if command in ("/clear", "/new"):
                agent.messages = [m for m in agent.messages if getattr(m, "role", None) == "system"]
                if session is not None:
                    session.start_new()
                    for message in agent.messages:
                        session.append(message)
                restore_compaction(agent.compactor, agent.messages,
                                   session.latest_custom("compaction") if session else None)
                agent._loaded_tools = []
                apply_tool_selection(agent)
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
                    if command == "/tree" and not rest.strip() and session and extensions:
                        await render_custom_entries(session, extensions)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"error: {exc}")
                continue
            if command == "/trust":
                if registry is None:
                    print("error: model registry unavailable")
                else:
                    try:
                        await trust_command(registry, ui)
                    except (OSError, ValueError) as exc:
                        print(f"error: {exc}")
                continue
            if command == "/bug":
                if registry is None:
                    print("error: model registry unavailable")
                    continue
                try:
                    await bug_report_command(rest, agent, session, registry, extensions, ui)
                except (OSError, ValueError, RuntimeError, ModelError, asyncio.TimeoutError, httpx.HTTPError) as exc:
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


IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


async def image_command(argument: str, registry: Optional[ModelRegistry],
                        renderer: Optional[CliRenderer] = None) -> None:
    """Handle /image [model] <prompt>: generate an image via openrouter-images."""
    parts = argument.strip()
    if not parts:
        raise ValueError("Usage: /image [model] <prompt>")
    model_id: Optional[str] = None
    first, separator, remainder = parts.partition(" ")
    if separator and "/" in first:
        model_id, parts = first, remainder
    api_key = None
    if registry is not None:
        api_key = await registry.auth.key("openrouter", None)
    model = resolve_image_model("openrouter", model_id, api_key=api_key)
    result = await generate_images(model, ImagesContext(input=[TextContent(text=parts)]))
    if result.stop_reason != "stop" and result.error_message:
        raise ValueError(result.error_message)
    if result.usage is not None:
        cost = result.usage.cost.get("total", 0.0) if isinstance(result.usage.cost, dict) else 0.0
        print(f"[{result.usage.total_tokens} tokens, ${cost:.4f}]")
    saved = 0
    for block in result.output:
        if isinstance(block, TextContent):
            print(block.text)
        elif isinstance(block, ImageContent):
            saved += 1
            extension = IMAGE_EXTENSIONS.get(block.mime_type, ".png")
            path = os.path.join(os.getcwd(), f"image-{saved}{extension}")
            with open(path, "wb") as handle:
                handle.write(base64.b64decode(block.data))
            print(f"saved: {path}")
            if renderer is not None:
                renderer.print_image_block(block)


def sync_block_images(agent: Agent, registry: ModelRegistry) -> None:
    """Apply the images.block_images setting to a running agent."""
    agent.block_images = bool(registry.config.settings.get("images", {}).get("block_images", False))


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


# Providers whose sign-in flow is OAuth unless an API key is stored manually.
OAUTH_DEFAULT_PROVIDERS = ("openai-codex", "github-copilot", "kimi-coding", "xai", "radius")
# Providers that also offer OAuth through `/login <provider> oauth`.
OAUTH_OPTIONAL_PROVIDERS = ("anthropic", "openai", "openrouter")


def login_provider_options(registry: ModelRegistry, oauth: bool) -> List[Tuple[str, str]]:
    """Provider choices for one authentication method of the `/login` picker."""
    if oauth:
        names = [name for name, provider in registry.providers.items()
                 if name in OAUTH_DEFAULT_PROVIDERS or name in OAUTH_OPTIONAL_PROVIDERS
                 or provider.get("oauth")]
    else:
        names = [name for name, provider in registry.providers.items()
                 if name != "faux" and name not in OAUTH_DEFAULT_PROVIDERS
                 and provider.get("auth_required", True)]
    return [(name, name) for name in sorted(names)]


async def authenticate(registry: ModelRegistry, provider: str, oauth: bool = False) -> None:
    provider = {"claude": "anthropic", "codex": "openai-codex", "copilot": "github-copilot",
        "chatgpt": "openai", "openai-chatgpt": "openai"}.get(provider, provider)
    if provider not in registry.providers:
        raise ModelError(f"Unknown provider: {provider}")
    if oauth or provider in OAUTH_DEFAULT_PROVIDERS:
        async def prompt(message):
            # Browser callbacks can cancel this prompt. Blocking getpass in
            # the default executor would keep asyncio.run() alive afterwards.
            return await PromptSession().prompt_async(message, is_password=True)
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
    if provider == "radius":
        try:
            count = await refresh_radius_catalog(registry)
            print(f"synced {count} models from the Radius gateway")
        except (OSError, ValueError, httpx.HTTPError) as exc:
            print(f"warning: could not sync the Radius model catalog: {describe_error(exc)}")


# Sign-in choices shared by the TUI picker and the plain numbered menu.
LOGIN_METHODS = [
    ("oauth", "Sign in with an account (OAuth)"),
    ("api_key", "Sign in with an API key"),
    ("custom", "Add custom API model"),
]


class TextWizardIO:
    """Numbered-prompt UI for the custom API wizard in plain terminals.

    Normal fields accept ":cancel" to abort; the password field does not parse
    it, so a key may legitimately contain that text (Ctrl+C/EOF cancel instead).
    """

    async def choose(self, title: str, values, default: Optional[str] = None) -> Optional[str]:
        print(title)
        for index, (value, label) in enumerate(values, 1):
            print(f"  {index}. {label}{'  [default]' if value == default else ''}")
        hint = f"Choose [1-{len(values)}]"
        if default is not None:
            hint += ", Enter for default"
        while True:
            raw = await asyncio.get_running_loop().run_in_executor(None, input, hint + ": ")
            text = raw.strip()
            if text.lower() == ":cancel":
                return None
            if not text:
                if default is not None:
                    return default
                print("error: choose a number")
                continue
            if text.isdigit() and 1 <= int(text) <= len(values):
                return values[int(text) - 1][0]
            print(f"error: enter a number between 1 and {len(values)}")

    async def text_input(self, title: str, default: str = "", password: bool = False,
                         allow_cancel_word: bool = True,
                         validate: Optional[Callable[[str], str]] = None) -> Optional[str]:
        while True:
            if password:
                raw = await asyncio.get_running_loop().run_in_executor(None, getpass.getpass, f"{title}: ")
            else:
                shown = f" [{default}]" if default else ""
                raw = await asyncio.get_running_loop().run_in_executor(None, input, f"{title}{shown}: ")
            if allow_cancel_word and raw.strip() == ":cancel":
                return None
            if validate is None:
                return raw
            try:
                return validate(raw)
            except ValueError as exc:
                print(f"error: {exc}")

    async def confirm(self, title: str, default: bool) -> Optional[bool]:
        result = await self.choose(title, [("yes", "Yes"), ("no", "No")], default="yes" if default else "no")
        return None if result is None else result == "yes"

    def say(self, message: str) -> None:
        print(message)


def _apply_custom_api_choices(registry: ModelRegistry, agent: Optional[Agent],
                              session: Optional[SessionManager], form: CustomApiForm,
                              use_now: Optional[bool], set_default: Optional[bool]) -> List[str]:
    """Set the default and switch models after a successful save.

    Returns the lines to show in the main UI. Post-save failures are reported
    separately from the save itself so the user never retries the save.
    """
    identifier = form.provider_id + "/" + form.model_id
    messages = [f"saved provider '{form.provider_id}' with model '{identifier}'"]
    if set_default:
        try:
            registry.config.save_many({"provider": form.provider_id, "model": form.model_id})
            messages.append(
                f"default model: {identifier} (project settings or --provider/--model still override it)")
        except (OSError, ValueError) as exc:
            messages.append(f"model saved, but setting the default failed: {describe_error(exc)}")
    else:
        messages.append("default model: unchanged")
    if agent is not None:
        if use_now:
            try:
                registry.reload()
                agent.model = registry.resolve(form.provider_id, form.model_id)
                if not agent.model.reasoning and agent.thinking_level != "off":
                    agent.thinking_level = "off"
                    if session:
                        session.append_thinking_change("off")
                if session:
                    session.append_model_change(agent.model.provider, agent.model.id)
                messages.append(f"model: {identifier}")
            except (ModelError, ValueError) as exc:
                messages.append(f"model saved, but switching failed: {describe_error(exc)}")
            except (OSError, RuntimeError) as exc:
                messages.append(f"warning: could not record the switch in the session: {describe_error(exc)}")
        else:
            messages.append("current model: unchanged")
    messages.append("note: the connection has not been verified yet; the first prompt sends the first real request")
    return messages


async def _custom_login_single_page(registry: ModelRegistry, agent: Optional[Agent],
                                    session: Optional[SessionManager], ui) -> None:
    """One-page flow for the enhanced terminal: validate, save, then report."""
    saver = CustomApiSaver(registry.config)

    def validate(values):
        return validate_form_values(values, registry.providers, registry.config)

    async def save(form: CustomApiForm, use_now: bool, set_default: bool) -> SaveResult:
        try:
            saver.save(registry, form)
        except ConcurrentConfigChange:
            try:
                registry.reload()
            except (ModelError, ValueError) as exc:
                raise SaveError(
                    f"configuration changed on disk and could not be reloaded: {describe_error(exc)}") from exc
            saver.adopt_current()
            # Keep the draft; the page shows fresh provider conflicts and asks
            # the user to press Save model again.
            raise
        return SaveResult(_apply_custom_api_choices(
            registry, agent, session, form, use_now, set_default))

    result = await ui.custom_api_form(validate, save)
    if result is None:
        print("custom API model login cancelled")
        return
    for message in result.messages:
        print(message)


async def custom_login_command(registry: ModelRegistry, agent: Optional[Agent] = None,
                               session: Optional[SessionManager] = None,
                               ui: Optional[TerminalUI] = None) -> None:
    """Run the add-custom-API-model wizard and apply the post-save choices.

    The enhanced terminal gets the single-page form; the plain REPL and
    ``--login custom`` keep the text wizard.
    """
    if RESERVED_PROVIDER_ID in registry.providers:
        raise ModelError("provider ID 'custom' is reserved for this login entry; "
                         "rename it in models.json to use the wizard")
    if agent is not None and agent.is_streaming:
        raise ValueError("cannot change the configuration while a request is running")
    if ui is not None:
        await _custom_login_single_page(registry, agent, session, ui)
        return
    io: WizardIO = TextWizardIO()
    form = await run_custom_api_wizard(registry.config, registry, io)
    if form is None:
        io.say("custom API model login cancelled")
        return
    use_now = await io.confirm("Use this model now?", default=True) if agent is not None else None
    set_default = await io.confirm("Set as default for future launches?", default=False)
    for message in _apply_custom_api_choices(registry, agent, session, form, use_now, set_default):
        io.say(message)


async def _choose_console(title: str, values):
    """Numbered input() selector for flows that run before the TUI exists."""
    print(title)
    for index, (label, _value) in enumerate(values, 1):
        print(f"  {index}. {label}")
    raw = await asyncio.get_running_loop().run_in_executor(
        None, input, f"Choose [1-{len(values)}]: ")
    try:
        index = int(raw.strip())
    except ValueError:
        return None
    return values[index - 1][1] if 1 <= index <= len(values) else None


async def trust_command(registry: ModelRegistry, ui) -> None:
    """Save a project trust decision for the current folder."""
    store = TrustStore(registry.config.agent_dir)
    cwd_path = str(registry.config.cwd)
    options = trust_options(cwd_path)
    chooser = ui.choose if ui else _choose_console
    selected = await chooser(
        "Trust project folder?\n" + cwd_path
        + "\n\nThis allows pi to load .pi settings and resources, and execute project extensions.",
        [(option["label"], option) for option in options])
    if selected is None:
        print("trust: no change")
        return
    if selected["updates"]:
        store.set_many(selected["updates"])
        print(f"project trust: {'trusted' if selected['trusted'] else 'not trusted'} (saved)")
    else:
        print(f"project trust: {'trusted' if selected['trusted'] else 'not trusted'} (this session only)")


async def bug_report_command(argument: str, agent: Agent, session: Optional[SessionManager],
                             registry: ModelRegistry, extensions, ui) -> None:
    """Run the /bug flow: hint, transcript/summary choice, upload or zip export."""
    chooser = ui.choose if ui else _choose_console
    hint = argument.strip()
    if not hint:
        hint = await asyncio.get_running_loop().run_in_executor(
            None, input, "What went wrong? (optional, Enter to skip): ")
    include_session = False
    if session is not None:
        include_session = await chooser("Include the session transcript?", [
            ("Yes, include the transcript", True), ("No", False)]) is True
    include_summary = False
    if not include_session:
        include_summary = await chooser(
            f"Attach a summary written by {agent.model.name or 'the current model'} instead?", [
                ("Yes, generate a summary", True), ("No", False)]) is True
    delivery = await chooser("Bug report", [
        ("Upload report", "upload"), ("Export as zip", "zip"), ("Cancel", "cancel")])
    if delivery in (None, "cancel"):
        print("bug report cancelled")
        return
    if delivery == "upload" and os.environ.get("PI_OFFLINE"):
        raise ValueError("Uploading bug reports requires online mode. Use Export as Zip instead.")
    if include_summary:
        print(f"writing summary with {agent.model.name or agent.model.id}...")
        summary = await generate_bug_report_summary(agent.messages, agent.model, hint, agent.thinking_level)
    else:
        summary = None
    try:
        project_settings = read_json(registry.config.cwd / ".pi" / "settings.json")
    except ValueError:
        project_settings = {}
    metadata = collect_bug_report_metadata(
        hint=hint, session_id=os.path.basename(session.session_file) if session else "no-session",
        cwd=str(registry.config.cwd), include_session=include_session,
        include_summary=summary is not None, message_count=len(agent.messages),
        model=agent.model, thinking_level=agent.thinking_level,
        extensions=[str(api.path) for api in extensions.extensions] if extensions else [],
        extension_errors=[], global_settings=registry.config.settings,
        project_settings=project_settings)
    crashes = read_crash_log(crash_log_path(registry.config.agent_dir))
    bundle = {
        "metadata": metadata,
        "diagnostics": collect_bug_report_diagnostics(session, crashes),
        "sessionJsonl": session.branch_jsonl() if include_session else None,
        "summary": summary,
    }
    if delivery == "upload":
        try:
            token = await registry.auth.key("radius", None)
        except (ValueError, OSError, httpx.HTTPError):
            token = None
        try:
            report_id = await upload_bug_report(bundle, token=token)
            print(f"bug report uploaded. report id: {report_id}")
        except (OSError, ValueError, httpx.HTTPError) as exc:
            fallback = await chooser(f"Upload failed: {exc}\n\nExport the report as a zip archive instead?", [
                ("Export as zip", "zip"), ("Cancel", "cancel")])
            if fallback != "zip":
                print("bug report cancelled")
                return
            delivery = "zip"
    if delivery == "zip":
        path = os.path.join(os.getcwd(), bug_report_archive_file_name(metadata["id"]))
        write_bug_report_archive(bundle, path)
        print(f"bug report exported to: {path}\nreport id: {metadata['id']}")
    if session is not None:
        session.append_custom(BUG_REPORT_CUSTOM_ENTRY_TYPE, {
            "id": metadata["id"], "createdAt": metadata["createdAt"], "hint": metadata["hint"],
            "sessionIncluded": include_session, "summaryIncluded": summary is not None,
            "delivery": delivery})
    if crashes:
        clear_crash_log(crash_log_path(registry.config.agent_dir))


def handle_session_command(command: str, argument: str, agent: Agent,
                           session: Optional[SessionManager]) -> None:
    if session is None:
        raise ValueError("Session persistence is disabled")
    if command == "/resume":
        if argument:
            agent.messages = session.resume(argument)
        else:
            for item in session.list_sessions():
                print(f"{item['id']}  {item['name']}  {item['messages']} messages")
    elif command == "/session":
        stats = session_stats(agent.messages, agent.model, session.session_file,
                              extra_usage=summary_usage_extra(session))
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
        restore_compaction(agent.compactor, agent.messages, session.latest_custom("compaction"))
        apply_tool_selection(agent)


async def render_custom_entries(session: SessionManager, extensions: ExtensionManager) -> None:
    for entry in session.get_branch():
        if entry.custom is None:
            continue
        custom_type = entry.custom["kind"]
        if entry.message is None:
            lines = await extensions.render_entry(custom_type, {
                "id": entry.id, "parentId": entry.parent_id,
                "customType": custom_type, "data": entry.custom["data"]}, {"expanded": True})
        else:
            lines = await extensions.render_message(custom_type, {
                "role": "custom", "customType": custom_type,
                "content": get_message_text(entry.message.content),
                "display": entry.custom["data"].get("display", True),
                "details": entry.custom["data"].get("details")}, {"expanded": True})
        if lines:
            print("\n".join(lines))


def _resolve_session_source(session: SessionManager, identifier: str) -> str:
    """A --fork / --session argument as a session file path: path or id prefix."""
    if os.path.isfile(identifier):
        return identifier
    sessions = session.list_sessions()
    matches = [item for item in sessions if item["id"] == identifier]
    if not matches:
        matches = [item for item in sessions if item["id"].startswith(identifier)]
    if len(matches) != 1:
        raise ValueError("Session ID is unknown or ambiguous")
    return matches[0]["path"]


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
    restore_compaction(agent.compactor, agent.messages, session.latest_custom("compaction"))


def sync_mcp_tools(agent: Agent, mcp: Optional[McpManager]) -> None:
    """Replace the agent's MCP tools with the manager's current set, honoring the
    run's --tools / --exclude-tools selection."""
    if mcp is None:
        return
    for tool in mcp.tools:
        tool._mcp_managed = True
    agent.tools[:] = [tool for tool in agent.tools if not getattr(tool, "_mcp_managed", False)] + list(mcp.tools)
    selection = getattr(agent, "_tool_selection", None)
    activate = getattr(agent, "_activate_searched_tools", None)
    if (activate and any(tool.exposure == "deferred" for tool in mcp.tools)
            and (selection is None or selection.keeps("tool_search"))
            and not any(tool.name == "tool_search" for tool in agent.tools)):
        agent.tools.append(create_tool_search_tool(agent.tools, activate))
    apply_tool_selection(agent)


def report_mcp_state(server: str, state: str, error: Optional[str] = None) -> None:
    """Print MCP connect failures as they happen; status stays visible via /mcp."""
    if state in ("failed", "login-required") and error:
        print(f"MCP {server}: {error}", file=sys.stderr)


def describe_error(exc: BaseException) -> str:
    """Exception text, falling back to the type name when the message is empty."""
    return str(exc) or type(exc).__name__


def radius_login_configured(registry: ModelRegistry) -> bool:
    """Whether Radius credentials exist.

    Discovery without credentials is a guaranteed failure (the TS runtime does
    not fetch the gateway catalog at all until Radius is configured), so startup
    skips it instead of warning on every non-Radius install.
    """
    if os.environ.get("RADIUS_API_KEY"):
        return True
    try:
        return registry.auth.get("radius") is not None
    except ValueError:
        return False


async def refresh_radius_models(registry: ModelRegistry) -> None:
    """Best-effort Radius gateway model discovery; failures keep the cached catalog."""
    if os.environ.get("PI_OFFLINE") or registry.config.settings.get("sync_catalog") is False:
        return
    if not radius_login_configured(registry):
        return
    try:
        await refresh_radius_catalog(registry)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"warning: Radius model discovery failed: {describe_error(exc)}", file=sys.stderr)


async def async_main(argv: Optional[List[str]] = None) -> int:
    # Redirected Windows streams can default to a legacy code page. Configure
    # text mode too, before startup diagnostics or model output are written.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    filtered_argv, extension_flag_values = split_extension_flags(raw_argv, build_parser())
    args = build_parser().parse_args(filtered_argv)
    if args.offline:
        os.environ["PI_OFFLINE"] = "1"
    if args.verbose:
        os.environ["PI_TIMING"] = "1"
        timings.ENABLED = True
    if args.no_session and (args.session or args.continue_session or args.resume or args.name or args.session_id):
        print("error: --no-session cannot be combined with session options", file=sys.stderr)
        return 2
    if args.session_id and (args.session or args.continue_session or args.resume):
        print("error: --session-id cannot be combined with --session, --continue or --resume",
              file=sys.stderr)
        return 2
    if args.fork and (args.session or args.continue_session or args.resume or args.no_session):
        print("error: --fork cannot be combined with --session, --continue, --resume or --no-session",
              file=sys.stderr)
        return 2
    if args.session_id and not SESSION_ID_PATTERN.fullmatch(args.session_id):
        print("error: --session-id must start and end with a letter or digit and may contain "
              "letters, digits, dots, dashes and underscores", file=sys.stderr)
        return 2
    cwd = os.getcwd()
    if args.export:
        # Standalone session export: no provider, registry or session state needed.
        output = next((arg for arg in args.prompt if not arg.startswith("@")), None)
        try:
            print(export_session_file(args.export, output))
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    file_args = [arg[1:] for arg in args.prompt if arg.startswith("@")]
    message_args = [arg for arg in args.prompt if not arg.startswith("@")]
    prompt_text = " ".join(message_args)
    prompt_images: Optional[List[Any]] = None
    if file_args:
        try:
            processed = await process_file_arguments(file_args, cwd=cwd, auto_resize_images=False)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # AgentSession-equivalent resize runs per model after hooks; the CLI
        # attachment pass only normalizes formats. Media files ride the same
        # attachment list; providers reject payloads they cannot accept.
        prompt_text = processed.text + prompt_text
        prompt_images = (processed.images + processed.media) or None
    if args.mode == "rpc" and args.prompt or args.mode == "json" and not args.prompt:
        print("error: rpc accepts prompts through stdin; json requires a prompt argument", file=sys.stderr)
        return 2
    if args.mode != "text" and (args.login or args.logout or args.list_models or args.sync_models or args.evals):
        print("error: authentication, model-list, sync and eval commands require text mode", file=sys.stderr)
        return 2

    timings.reset_timings()
    # Protocol modes protect their stdout stream from stray print() output.
    if args.mode in ("json", "rpc"):
        take_over_stdout()
    has_ui = args.mode == "text" and not message_args and not args.print_mode and sys.stdin.isatty()

    try:
        registry = ModelRegistry(Configuration(cwd, args.agent_dir or ""))
        CRASH_CONTEXT.update({"cwd": cwd, "agent_dir": str(registry.config.agent_dir)})
        if args.sync_models:
            providers = await fetch_catalog()
            save_cache(registry.config.agent_dir, providers)
            radius_count = 0
            try:
                radius_count = await refresh_radius_catalog(registry)
            except (OSError, ValueError, httpx.HTTPError) as exc:
                print(f"warning: Radius model discovery failed: {describe_error(exc)}", file=sys.stderr)
            registry.reload()
            total = sum(len(entry["models"]) for entry in providers.values())
            print(f"synced {total} models across {len(providers)} providers")
            if radius_count:
                print(f"synced {radius_count} Radius models from the gateway")
            return 0
        if args.login:
            if args.login == "custom":
                if args.oauth:
                    print("error: --login custom cannot be combined with --oauth", file=sys.stderr)
                    return 2
                if RESERVED_PROVIDER_ID in registry.providers:
                    print("error: provider ID 'custom' is reserved for this login entry; "
                          "rename it in models.json to use the wizard", file=sys.stderr)
                    return 2
                if not sys.stdin.isatty():
                    print("error: --login custom requires an interactive terminal", file=sys.stderr)
                    return 2
                await custom_login_command(registry)
                return 0
            await authenticate(registry, args.login, args.oauth)
            return 0
        if args.logout:
            registry.auth.logout(args.logout)
            return 0
        # Startup model discovery (TS refreshes on every startup; PI_OFFLINE and
        # sync_catalog: false skip it, and failures fall back to the cache).
        await refresh_radius_models(registry)
        if args.list_models:
            print_models(registry)
            return 0
        if args.oauth:
            raise ModelError("--oauth requires --login")
        timings.time("registry")
        # The trust default must come from user settings only: an untrusted
        # project must not be able to set default_project_trust=always.
        global_settings = read_json(registry.config.agent_dir / "settings.json")
        trusted = await resolve_project_trusted(
            cwd, TrustStore(registry.config.agent_dir), _choose_console if has_ui else None,
            settings=global_settings, override=args.approve, has_ui=has_ui,
            notify=None if has_ui else (lambda message: print(f"warning: {message}", file=sys.stderr)))
        registry.config.project_trusted = trusted
        # Re-merge settings so untrusted project settings are not applied.
        registry.config.reload()
        registry.reload()
        if args.models:
            registry.model_scope = [entry.strip() for entry in args.models.split(",") if entry.strip()]
            if not registry.model_scope:
                raise ModelError("--models requires at least one provider/model pattern")
        try:
            themes = load_themes(registry.config.agent_dir,
                                 extra_paths=[Path(path).expanduser() for path in args.theme or []],
                                 no_themes=args.no_themes)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.use_theme:
            # Runtime override only: unlike /theme, --use-theme does not persist.
            if args.use_theme not in themes:
                print(f"error: unknown theme: {args.use_theme}. Available: {', '.join(sorted(themes))}",
                      file=sys.stderr)
                return 2
            registry.config.settings["theme"] = args.use_theme
        resources = ResourceCatalog(
            registry.config,
            skill_paths=[Path(path).expanduser() for path in args.skill or []],
            prompt_paths=[Path(path).expanduser() for path in args.prompt_template or []],
            no_skills=args.no_skills, no_prompts=args.no_prompt_templates)
        report_diagnostics(deduplicate_diagnostics([*settings_diagnostics(registry.config.errors),
            *[Diagnostic("warning", message) for message in resources.diagnostics]]))
        agent = make_agent(args, cwd, registry, resources)
        timings.time("agent")
        if has_ui:
            crash = take_unnotified_crash(crash_log_path(registry.config.agent_dir))
            if crash:
                print(f"warning: the previous run crashed: {crash['message']} (run /bug to report it)",
                      file=sys.stderr)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session: Optional[SessionManager] = None
    if not args.no_session:
        session = SessionManager(session_dir=args.session_dir or registry.config.settings.get("session_dir"), cwd=cwd)
        resumed = False
        if args.resume:
            if not has_ui:
                print("error: --resume requires an interactive terminal; use --session <path|id> instead",
                      file=sys.stderr)
                return 2
            items = session.list_sessions()
            chosen = await _choose_console(
                "Resume session",
                [(f"{item['name'] or item['id']} ({item['messages']} messages)", item["path"]) for item in items],
            ) if items else None
            if chosen is None:
                print("No session selected")
                return 0
            agent.messages = session.load(chosen)
            resumed = True
        elif args.session:
            agent.messages = session.resume(args.session)
            resumed = True
        elif args.fork:
            if args.session_id and any(item["id"] == args.session_id for item in session.list_sessions()):
                print(f"error: a session with id '{args.session_id}' already exists", file=sys.stderr)
                return 1
            try:
                source = _resolve_session_source(session, args.fork)
                agent.messages = session.fork_from(source, session_id=args.session_id)
            except (OSError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            resumed = True
        elif args.session_id:
            matches = [item for item in session.list_sessions() if item["id"] == args.session_id]
            if matches:
                agent.messages = session.load(matches[0]["path"])
                resumed = True
            else:
                print(f"warning: no project session found with id '{args.session_id}'; "
                      f"creating a new session with that id", file=sys.stderr)
                session.start_new(args.session_id)
                for message in agent.messages:
                    session.append(message)
        elif args.continue_session:
            latest = session.latest_session_file()
            if latest is None:
                print("no previous session found", file=sys.stderr)
                session.start_new()
                for message in agent.messages:
                    session.append(message)
            else:
                agent.messages = session.load(latest)
                resumed = True
                print(f"continued session: {os.path.basename(latest)}", file=sys.stderr)
        else:
            session.start_new()
            for message in agent.messages:
                session.append(message)
        if args.name:
            session.rename(args.name)
        if resumed:
            restore_session_settings(agent, session, registry)
            update = SystemMessage(content="", timestamp=0,
                                   sections={"skills": resources.system_prompt(file_read_tool=skill_file_read_tool(agent)) or None})
            agent.messages.append(update)
            session.append(update)

        def persist_summary(summary: str, usage=None) -> None:
            session.append_compaction(summary, checkpoint=prefix_checkpoint(agent.compactor.prefix), usage=usage)

        agent.compactor.on_summary = persist_summary
        CRASH_CONTEXT["session_file"] = session.session_file
        if is_opencode_like(agent.model):
            apply_attribution(agent.model, registry.config.settings,
                              session_id=os.path.basename(session.session_file or ""))

    renderer = CliRenderer(settings=registry.config.settings)
    agent._tool_search_session = session

    # Prompt cache warming (F-CA-13): off by default; streaming/idle keep the
    # provider-side cache alive with one-token replays of the last request.
    warmer = None
    warming_mode = registry.config.settings.get("cache_warming", "off")
    if warming_mode in ("streaming", "idle"):
        from .cache_warmer import CacheWarmer

        warmer = CacheWarmer(get_default_stream_fn())
        warmer.mode = warming_mode
        agent.on_real_request = warmer.capture
        agent.subscribe(lambda event: warmer.on_agent_settled()
                        if getattr(event, "type", "") == "agent_settled" else None)

    # --no-mcp disables built-in MCP support: no servers connect, no MCP tools,
    # and extensions see no MCP manager to register servers on.
    mcp = None if args.no_mcp else McpManager(registry.config)
    extensions = ExtensionManager(registry.config, agent, session, registry=registry, mcp=mcp,
                                  flag_values=extension_flag_values,
                                  extra_paths=[Path(path).expanduser() for path in args.extension or []],
                                  no_discovery=args.no_extensions)
    renderer.extensions = extensions
    try:
        if mcp is not None:
            # Connects run in the background so a slow server cannot hold startup;
            # tools arrive through on_tools_changed as each server lands.
            mcp.on_tools_changed = lambda server: sync_mcp_tools(agent, mcp)
            mcp.on_state_changed = report_mcp_state
            await mcp.reload(wait=False)
            sync_mcp_tools(agent, mcp)
        try:
            await extensions.reload()
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        registry.register_extension_providers(extensions.all_providers())
        apply_tool_selection(agent)
        # Codemode runs through the selection like any other tool: --tools without a
        # codemode entry and --no-tools / --exclude-tools codemode all leave it out.
        codemode_settings = registry.config.settings.get("codemode", {})
        selection = getattr(agent, "_tool_selection", None)
        if codemode_settings.get("enabled") and (selection is None or selection.keeps("codemode")):
            agent.tools.append(create_codemode_tool(
                agent.tools, session, codemode_settings,
                execute_tool=execute_nested_tool, model_registry=registry))
        timings.time("extensions")
        if args.evals:
            passed = await run_evals_file(agent, renderer, session, args.evals, mcp)
            timings.print_timings()
            return 0 if passed else 1
        timings.print_timings()
        if args.mode == "rpc":
            server = RpcServer(agent, JsonWriter(output=raw_stdout()), session, registry, resources, extensions)
            extensions.set_ui_sink(server.ui_sink)
            await server.serve()
            return 0
        if args.mode == "json":
            writer = WireJsonWriter(output=raw_stdout(), session=session)
            try:
                await run_prompt(agent, prompt_text, writer, session, resources, extensions, mcp, images=prompt_images)
            except Exception as exc:
                await writer.write({"type": "run_error", "error": str(exc)})
                return 1
            finally:
                await writer.write({"type": "agent_settled", "aborted": agent.abort_requested})
            return 0
        if args.print_mode:
            # -p forces one-shot output even without a prompt argument: piped
            # stdin becomes the prompt, mirroring the TS --print consumption.
            text = prompt_text
            if not text and not sys.stdin.isatty():
                text = sys.stdin.read().strip()
            if not text:
                print("error: --print requires a prompt argument or piped stdin", file=sys.stderr)
                return 2
            await run_prompt(agent, text, renderer, session, resources, extensions, mcp, images=prompt_images)
            return 0
        if message_args:
            await run_prompt(agent, prompt_text, renderer, session, resources, extensions, mcp, images=prompt_images)
            return 0
        ui = None
        if args.tui in ("on", "fullscreen") or args.tui == "auto" and sys.stdin.isatty() and sys.stdout.isatty():
            def commands():
                names = [line.strip().split()[0] for line in SLASH_HELP.splitlines() if line.strip().startswith("/")]
                names.extend("/" + name for name in extensions.commands)
                names.extend("/" + name for name in resources.prompts)
                if registry.config.settings.get("enable_skill_commands", True):
                    names.extend("/skill:" + name for name in resources.skills)
                return names
            ui = TerminalUI(agent, commands, registry.config.settings,
                            extra_bindings=extensions.all_keybindings(),
                            themes=themes, full_screen=args.tui == "fullscreen")
            ui.markdown_transform = lambda text: extensions.transform_markdown(text, "user")
            renderer.ui = ui
            renderer.full_screen = ui.full_screen
            extensions.set_custom_host(ui.custom_view, size=lambda: (ui.width(), ui.height()))
            extensions.set_editor_host(ui.set_custom_editor)
        if ui and ui.full_screen:
            extensions.set_ui_sink(ui.ui_request)
        with redirect_stdout(ui) if ui and ui.full_screen else nullcontext():
            await interactive_loop(agent, renderer, session, registry, resources, mcp, extensions, ui,
                                   PackageManager(registry.config), initial=(prompt_text, prompt_images or []))
        return 0
    finally:
        if warmer is not None:
            await warmer.close()
        await extensions.close()
        if mcp is not None:
            await mcp.close()


def _route_durable_command(argv: List[str]) -> Optional[int]:
    """Delegate the routed-protocol commands, so installed ``pi <command>`` works too."""
    if not argv:
        return None
    first = argv[0]
    if first == "durable":
        from ..durable.cli import main as routed_main

        return routed_main(argv[1:])
    if first == "durable-worker":
        from ..durable.worker import main as routed_main

        return routed_main(argv[1:])
    if first == "server":
        from ..durable.server import main as routed_main

        return routed_main(argv[1:])
    if first == "mcp":
        from .mcp_cli import main as mcp_main

        return mcp_main(argv[1:])
    return None


def main(argv: Optional[List[str]] = None) -> int:
    forwarded = list(sys.argv[1:]) if argv is None else list(argv)
    routed = _route_durable_command(forwarded)
    if routed is not None:
        return routed
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        print()
        return 130
    except (OSError, ValueError, ModelError, asyncio.TimeoutError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - persist the crash, then surface it
        record_crash("uncaught_exception", exc,
                     cwd=CRASH_CONTEXT.get("cwd", os.getcwd()),
                     session_file=CRASH_CONTEXT.get("session_file"),
                     agent_dir=CRASH_CONTEXT.get("agent_dir"))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        restore_stdout()


if __name__ == "__main__":
    sys.exit(main())
