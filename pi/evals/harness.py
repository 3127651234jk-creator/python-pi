"""Pi coding-agent eval harness.

Port of ``packages/evals/src/harness.ts`` onto the Python port's substrate:
instead of ``createAgentSessionFromServices`` from pi-coding-agent it assembles
``pi.agent.Agent`` + ``SessionManager`` + ``ModelRegistry`` the same way the
Python CLI does, inside an isolated temporary root.

The documentation variant machinery (with_docs / without_docs), the isolated
process environment, the POSIX tool sandbox, transcript normalization, and
usage accounting all follow the TypeScript implementation. Input steps accept
``{"type": "reload"}`` like the TypeScript harness: the Python agent has no
extension runner to rebuild, so a reload re-reads the isolated agent
directory's model configuration and hands the refreshed registry to the
``output`` grading hook. The TypeScript Vite-specific transformed-module
protection has no Python equivalent and is omitted; the container image keeps
eval sources root-only instead.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..ai.models import ModelError
from ..ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage
from ..agent.agent import Agent
from ..agent.types import AgentTool, MessageEndEvent
from ..coding_agent.auth import AuthStore
from ..coding_agent.config import Configuration
from ..coding_agent.model_registry import ModelRegistry
from ..coding_agent.session import SessionManager
from ..coding_agent.system_prompt import SystemPromptOptions, build_system_prompt
from ..coding_agent.tools import create_default_tools, tool_prompt_contributions
from ..coding_agent.usage import session_stats
from ..util import AbortSignal
from .framework import (
    HarnessRunArgs,
    attach_harness_run_to_error,
    create_harness,
    current_system_prompt,
    normalize_harness_run,
    normalize_record,
)
from .plan import DocumentationVariant
from .report import PI_SESSION_SNAPSHOT_ARTIFACT

DOCUMENTATION_EVAL_TOOLS = ["read", "write", "edit", "grep", "find", "ls"]


class AggregateError(Exception):
    """Several failures reported together (run error plus cleanup errors)."""

    def __init__(self, errors: List[BaseException], message: str) -> None:
        self.errors = list(errors)
        super().__init__(f"{message}: " + "; ".join(str(error) for error in self.errors))


@dataclass
class ModelSelection:
    provider: str
    id: str


PiCodingAgentInput = Any  # str or a list of {"type": "prompt"|"reload", ...} steps


@dataclass
class PiCodingAgentHarnessOptions:
    """Harness options; ``output`` upgrades the result to a typed output."""

    name: Optional[str] = None
    model: Optional[ModelSelection] = None
    no_tools: bool = False
    tools: Optional[List[str]] = None
    custom_tools: Optional[List[AgentTool]] = None
    workspace_files: Optional[Mapping[str, str]] = None
    transform_system_prompt: Optional[Callable[[str], str]] = None
    expected_pi_documentation: Optional[bool] = None
    output: Optional[Callable[..., Any]] = None


def resolve_model_selection(explicit_model: Optional[ModelSelection],
                            environment: Optional[Mapping[str, str]] = None) -> ModelSelection:
    """Prefer an explicit harness model, then PI_PROVIDER/PI_MODEL defaults."""
    environment = environment if environment is not None else os.environ
    provider = (explicit_model.provider if explicit_model else None) or (environment.get("PI_PROVIDER") or "").strip()
    model_id = (explicit_model.id if explicit_model else None) or (environment.get("PI_MODEL") or "").strip()
    if not provider or not model_id:
        raise ValueError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL as defaults.")
    return ModelSelection(provider=provider, id=model_id)


def get_agent_dir() -> str:
    """The host agent directory holding the shared auth.json."""
    return os.environ.get("PI_PY_AGENT_DIR") or str(Path.home() / ".pi-python")


def apply_isolated_environment(home: str, agent_dir: str) -> Callable[[], None]:
    """Point the process at an isolated home/agent dir and drop PI_EVAL_* vars."""
    overrides = {"HOME": home, "USERPROFILE": home, "PI_PY_AGENT_DIR": agent_dir}
    previous: Dict[str, Optional[str]] = {}
    for name in list(os.environ):
        if name.startswith("PI_EVAL_"):
            previous[name] = os.environ[name]
            del os.environ[name]
    for name, value in overrides.items():
        if name not in previous:
            previous[name] = os.environ.get(name)
        os.environ[name] = value

    def restore() -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return restore


def parse_sandbox_id(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        identifier = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None
    if identifier < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return identifier


def resolve_sandbox_identity() -> Optional[Dict[str, int]]:
    """PI_EVAL_SANDBOX_UID/GID must be set together on POSIX hosts."""
    uid = parse_sandbox_id("PI_EVAL_SANDBOX_UID")
    gid = parse_sandbox_id("PI_EVAL_SANDBOX_GID")
    if uid is None and gid is None:
        return None
    if uid is None or gid is None:
        raise ValueError("Set both PI_EVAL_SANDBOX_UID and PI_EVAL_SANDBOX_GID, or neither.")
    return {"uid": uid, "gid": gid}


def _chown_tree(path: str, identity: Dict[str, int]) -> None:
    stats = os.lstat(path)
    if stat.S_ISDIR(stats.st_mode):
        for entry in os.listdir(path):
            _chown_tree(os.path.join(path, entry), identity)
    os.chown(path, identity["uid"], identity["gid"], follow_symlinks=False)


def enter_tool_sandbox(root: str, identity: Optional[Dict[str, int]]) -> None:
    """Permanently drop to the unprivileged eval user (POSIX, root start only)."""
    if not identity:
        return
    if not all(hasattr(os, name) for name in ("getuid", "geteuid", "setuid", "setgid", "setgroups")):
        raise OSError("The eval filesystem sandbox requires POSIX user APIs.")
    if os.getuid() != 0 or os.geteuid() != 0:
        raise PermissionError("The eval runner must start as root before entering the unprivileged tool sandbox.")
    _chown_tree(root, identity)
    os.setgroups([])
    os.setgid(identity["gid"])
    os.setuid(identity["uid"])
    if os.getuid() != identity["uid"] or os.geteuid() != identity["uid"]:
        raise PermissionError("Failed to enter the unprivileged eval tool sandbox.")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content if isinstance(block, TextContent))


def to_transcript_events(messages: List[Any]) -> List[Dict[str, Any]]:
    """Normalize a Python transcript into the interchange event shape."""
    events: List[Dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role == "user":
            events.append({"type": "message", "role": "user", "content": _content_text(message.content)})
        elif role == "assistant":
            text = _content_text(message.content)
            if text:
                events.append({"type": "message", "role": "assistant", "content": text})
            for part in message.content:
                if isinstance(part, ToolCall):
                    events.append({
                        "type": "tool_call",
                        "id": part.id,
                        "name": part.name,
                        "arguments": normalize_record(part.arguments),
                    })
        elif role == "toolResult" and isinstance(message, ToolResultMessage):
            all_text = all(isinstance(part, TextContent) for part in message.content)
            events.append({
                "type": "tool_result",
                "toolCallId": message.tool_call_id,
                "name": message.tool_name,
                "content": _content_text(message.content) if all_text else [asdict(part) for part in message.content],
                **({"error": {"message": _content_text(message.content) or "Tool failed"}} if message.is_error else {}),
            })
    return events


def seed_workspace(workspace: str, files: Optional[Mapping[str, str]]) -> None:
    """Materialize workspace fixtures, rejecting absolute or escaping paths."""
    for name, content in (files or {}).items():
        if not name or os.path.isabs(name):
            raise TypeError(f"Invalid workspace fixture path: {name}")
        path = Path(workspace).resolve() / name
        path_from_workspace = os.path.relpath(str(path), workspace)
        if path_from_workspace == ".." or path_from_workspace.startswith(".." + os.sep) or os.path.isabs(path_from_workspace):
            raise TypeError(f"Workspace fixture escapes the workspace: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        os.chmod(path, 0o600)


async def prompt_agent(agent: Agent, input_text: str, signal: Optional[AbortSignal]) -> str:
    """Run one prompt and return the final assistant text."""
    if signal is not None:
        signal.throw_if_aborted()
    previous_message_count = len(agent.messages)
    await agent.prompt(input_text)
    assistant = next(
        (message for message in reversed(agent.messages[previous_message_count:])
         if isinstance(message, AssistantMessage)),
        None,
    )
    if assistant is None:
        raise RuntimeError("Agent run completed without an assistant message.")
    if assistant.stop_reason not in ("stop", "toolUse"):
        raise RuntimeError(assistant.error_message
                           or f"Agent run ended with unexpected stop reason: {assistant.stop_reason}.")
    output = _content_text(assistant.content)
    if not output and assistant.stop_reason == "stop":
        raise RuntimeError("Agent run produced no assistant text.")
    return output


def verify_system_prompt(system_prompt: str, *, name: Optional[str],
                         expected_pi_documentation: Optional[bool]) -> str:
    """Fail closed when the prompt does not match the eval variant."""
    if expected_pi_documentation is None:
        return system_prompt
    if "\n<rules>\n" not in system_prompt:
        raise RuntimeError(f"Pi system prompt lost its rules in the {name} eval variant.")
    has_documentation = "\n<docs>\nPi documentation (read only" in system_prompt
    if has_documentation != expected_pi_documentation:
        raise RuntimeError(f"Pi system prompt does not match the {name} eval variant.")
    return system_prompt


def exclude_pi_documentation(default_prompt: str) -> str:
    """Strip the Pi documentation-routing section from the default prompt."""
    documentation_start_marker = "\n<docs>\n"
    documentation_end_marker = "\n</docs>"
    documentation_start = default_prompt.find(documentation_start_marker)
    if documentation_start == -1:
        raise RuntimeError("Default Pi system prompt has no Pi documentation section.")
    documentation_end = default_prompt.find(documentation_end_marker, documentation_start)
    if documentation_end == -1:
        raise RuntimeError("Default Pi system prompt has no complete Pi documentation section.")
    cwd_start = default_prompt.rfind("\n<cwd>\n")
    if cwd_start < documentation_end:
        raise RuntimeError("Default Pi system prompt has no working-directory section.")
    return (default_prompt[:documentation_start]
            + default_prompt[documentation_end + len(documentation_end_marker):])


def resolve_documentation_variant(value: Optional[str] = None) -> DocumentationVariant:
    value = os.environ.get("PI_EVAL_VARIANT") if value is None else value
    if value in ("without_docs", "with_docs"):
        return value
    raise TypeError('PI_EVAL_VARIANT must be "without_docs" or "with_docs".')


def _credential_source_env(selection: ModelSelection, api_key: str) -> Optional[str]:
    """The environment variable an api-key credential was resolved from."""
    candidates = [f"{selection.provider.upper().replace('-', '_')}_API_KEY", "PI_PY_API_KEY"]
    for candidate in candidates:
        if candidate and os.environ.get(candidate) == api_key:
            return candidate
    return None


async def _resolve_eval_model(workspace: str, agent_dir: str, host_agent_dir: str, selection: ModelSelection):
    """Build the registry in the isolated agent dir and resolve the model."""
    host_auth = Path(host_agent_dir) / "auth.json"
    stored_credential = None
    try:
        stored_credential = AuthStore(host_auth).get(selection.provider)
    except (ValueError, OSError):
        stored_credential = None
    registry = ModelRegistry(Configuration(workspace, agent_dir))
    if stored_credential:
        registry.auth.set(selection.provider, stored_credential)
    try:
        model = registry.resolve(selection.provider, selection.id)
    except ModelError as error:
        raise RuntimeError(f"Eval model not found: {selection.provider}/{selection.id}") from error
    try:
        api_key = await model.get_api_key(selection.provider)
    except ModelError as error:
        raise RuntimeError(
            f"Eval model has no configured authentication: {selection.provider}/{selection.id}") from error

    async def fixed_api_key(_provider: str) -> str:
        return api_key

    model.get_api_key = fixed_api_key
    return registry, model, api_key, stored_credential


def _create_agent(workspace: str, model: Any, options: PiCodingAgentHarnessOptions):
    """Assemble the agent exactly like the CLI does, with eval-selected tools."""
    base_tools = create_default_tools(workspace)
    custom_tools = list(options.custom_tools or [])
    if options.no_tools:
        tools: List[AgentTool] = []
    elif options.tools is not None:
        known = {tool.name: tool for tool in base_tools}
        for tool in custom_tools:
            known.setdefault(tool.name, tool)
        missing = [name for name in options.tools if name not in known]
        if missing:
            raise ValueError(f"Unknown eval tools: {', '.join(missing)}")
        tools = [known[name] for name in options.tools]
        custom_tools = []
    else:
        tools = base_tools
    tools.extend(custom_tools)
    snippets, guidelines = tool_prompt_contributions(tools)
    prompt_options = SystemPromptOptions(
        cwd=workspace,
        selected_tools=[tool.name for tool in tools],
        tool_snippets=snippets,
        tool_guidelines=guidelines,
        include_pi_documentation=True,
    )
    default_prompt = build_system_prompt(prompt_options)
    forced_prompt = None
    if options.transform_system_prompt:
        forced_prompt = options.transform_system_prompt(default_prompt)
    agent = Agent(model=model, tools=tools, system_prompt=forced_prompt or default_prompt, thinking_level="off")
    return agent, forced_prompt


def _has_pricing(model: Any) -> bool:
    costs = model.costs
    return any(getattr(costs, field) > 0 for field in ("input", "output", "cache_read", "cache_write"))


async def run_pi_coding_agent(args: HarnessRunArgs,
                              options: PiCodingAgentHarnessOptions) -> Dict[str, Any]:
    """Run the input steps against an isolated agent session."""
    started_at = time.perf_counter()
    if args.signal is not None:
        args.signal.throw_if_aborted()
    selection = resolve_model_selection(options.model)
    host_agent_dir = get_agent_dir()
    sandbox_identity = resolve_sandbox_identity()
    if sandbox_identity:
        if not all(hasattr(os, name) for name in ("getuid", "geteuid", "setuid", "setgid", "setgroups")):
            raise OSError("The eval filesystem sandbox requires POSIX user APIs.")
        if os.getuid() != 0 or os.geteuid() != 0:
            raise PermissionError("The eval runner must start as root before entering the unprivileged tool sandbox.")
    root = tempfile.mkdtemp(prefix="pi-eval-")
    workspace = os.path.join(root, "workspace")
    isolated_home = os.path.join(root, "home")
    agent_dir = os.path.join(isolated_home, ".pi-python")
    forced_prompt: Optional[str] = None

    session_manager: Optional[SessionManager] = None
    agent: Optional[Agent] = None
    result: Optional[Dict[str, Any]] = None
    run_diagnostics: Optional[Dict[str, Any]] = None
    run_error: Optional[BaseException] = None
    cleanup_errors: List[BaseException] = []
    hidden_credential_environment: Optional[Dict[str, str]] = None
    restore_environment = apply_isolated_environment(isolated_home, agent_dir)
    try:
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(agent_dir, exist_ok=True)
        seed_workspace(workspace, options.workspace_files)
        registry, model, api_key, stored_credential = await _resolve_eval_model(
            workspace, agent_dir, host_agent_dir, selection)
        if sandbox_identity:
            try:
                os.remove(os.path.join(agent_dir, "auth.json"))
            except FileNotFoundError:
                pass
            if api_key and not stored_credential:
                source = _credential_source_env(selection, api_key)
                if source and os.environ.get(source):
                    hidden_credential_environment = {"name": source, "value": os.environ[source]}
                    del os.environ[source]

        agent, forced_prompt = _create_agent(workspace, model, options)
        if args.signal is not None:
            args.signal.throw_if_aborted()
        session_manager = SessionManager(session_dir=os.path.join(root, "sessions"), cwd=workspace)
        session_manager.start_new()
        args.set_artifact("runId", Path(session_manager.session_file).stem)
        agent.compactor.on_summary = (
            lambda summary, usage=None: session_manager.append_compaction(summary, usage=usage))
        for message in list(agent.messages):
            session_manager.append(message)

        def record(event: Any) -> None:
            if isinstance(event, MessageEndEvent):
                session_manager.append(event.message)

        unsubscribe = agent.subscribe(record)
        enter_tool_sandbox(root, sandbox_identity)
        response: Optional[str] = None
        steps: List[Dict[str, Any]] = (
            [{"type": "prompt", "content": args.input}] if isinstance(args.input, str) else list(args.input)
        )
        try:
            for step in steps:
                kind = step.get("type")
                if kind == "reload":
                    # The TypeScript reload rebuilds extensions and settings; the
                    # Python agent has neither, so a reload re-reads the agent
                    # directory's model configuration (models.json/auth.json),
                    # which is what provider/model evals observe afterwards. A
                    # broken configuration does not fail the run: the registry
                    # keeps its last good state and the output hook's own reload
                    # reports the failure as a structured probe error.
                    try:
                        registry.reload()
                    except (ModelError, ValueError):
                        pass
                    continue
                if kind != "prompt":
                    raise ValueError(f"Unknown eval input step type: {kind!r}")
                response = await prompt_agent(agent, step["content"], args.signal)
        finally:
            unsubscribe()
        if response is None:
            raise ValueError("Pi eval input must include at least one prompt step.")
        # A forced prompt is not recorded in the transcript, so use the one the
        # transform produced; otherwise the replayed transcript prompt is what
        # the provider received.
        system_prompt = forced_prompt or current_system_prompt(agent.messages)
        stats = session_stats(agent.messages, model)
        usage_metadata: Dict[str, Any] = {
            "cacheReadTokens": stats["tokens"]["cache_read"],
            "cacheWriteTokens": stats["tokens"]["cache_write"],
        }
        if _has_pricing(model):
            usage_metadata["estimatedCostUsd"] = stats["cost"]
        run_diagnostics = {
            "events": to_transcript_events(agent.messages),
            "metadata": {"systemPromptSha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()},
            "usage": {
                "provider": model.provider,
                "model": model.id,
                "inputTokens": stats["tokens"]["input"],
                "outputTokens": stats["tokens"]["output"],
                "totalTokens": stats["tokens"]["total"],
                "toolCalls": stats["tool_calls"],
                "metadata": usage_metadata,
            },
        }
        verify_system_prompt(system_prompt, name=options.name,
                             expected_pi_documentation=options.expected_pi_documentation)
        output = response
        if options.output is not None:
            output = options.output(response=response, session=agent,
                                    system_prompt=system_prompt, agent_dir=agent_dir, registry=registry)
            if inspect.isawaitable(output):
                output = await output
        result = {"output": output, **run_diagnostics}
    except BaseException as error:  # noqa: BLE001 - the run records failures
        run_error = error
    finally:
        if session_manager is not None:
            session_path = session_manager.session_file
            if not session_path or not os.path.exists(session_path):
                cleanup_errors.append(RuntimeError("Pi eval produced no session file."))
            else:
                try:
                    args.set_artifact(PI_SESSION_SNAPSHOT_ARTIFACT,
                                      Path(session_path).read_text(encoding="utf-8"))
                except OSError as error:
                    cleanup_errors.append(error)
        try:
            shutil.rmtree(root, ignore_errors=True)
        except OSError as error:
            cleanup_errors.append(error)
        restore_environment()
        if hidden_credential_environment:
            os.environ[hidden_credential_environment["name"]] = hidden_credential_environment["value"]

    failure: Optional[BaseException] = run_error
    if run_error is not None and cleanup_errors:
        failure = AggregateError([run_error, *cleanup_errors], "Agent run failed and cleanup also failed.")
    elif len(cleanup_errors) == 1:
        failure = cleanup_errors[0]
    elif cleanup_errors:
        failure = AggregateError(cleanup_errors, "Agent cleanup failed.")
    if failure is not None:
        if run_diagnostics:
            partial_run = normalize_harness_run(args.input, {
                **run_diagnostics,
                "errors": [failure],
                "timings": {"totalMs": (time.perf_counter() - started_at) * 1000},
            })
            raise attach_harness_run_to_error(failure, partial_run)
        raise failure
    if result is None:
        raise RuntimeError("Pi eval completed without a result.")
    return {**result, "timings": {"totalMs": (time.perf_counter() - started_at) * 1000}}


def create_pi_coding_agent_harness(options: Optional[PiCodingAgentHarnessOptions] = None):
    """Harness adapter that runs the isolated Python pi coding agent."""
    options = options or PiCodingAgentHarnessOptions()

    async def run(args: HarnessRunArgs) -> Dict[str, Any]:
        return await run_pi_coding_agent(args, options)

    return create_harness(name=options.name or "pi-coding-agent", run=run)


def create_pi_documentation_eval_harness(options: Optional[PiCodingAgentHarnessOptions] = None):
    """Documentation-lift harness: fixed tools, variant-driven prompt and check."""
    options = options or PiCodingAgentHarnessOptions()
    if os.environ.get("PI_EVAL_CONTAINER") != "1" or not resolve_sandbox_identity():
        raise RuntimeError("Documentation evals must run in the isolated container sandbox.")
    variant = resolve_documentation_variant()
    return create_pi_coding_agent_harness(PiCodingAgentHarnessOptions(
        name=variant,
        model=options.model,
        no_tools=options.no_tools,
        tools=options.tools if options.tools is not None else list(DOCUMENTATION_EVAL_TOOLS),
        custom_tools=options.custom_tools,
        workspace_files=options.workspace_files,
        expected_pi_documentation=variant == "with_docs",
        output=options.output,
        transform_system_prompt=exclude_pi_documentation if variant == "without_docs" else None,
    ))
