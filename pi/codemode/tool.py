"""Codemode tool: agent-tool adapter over the codemode host runtime.

TS counterpart: ``packages/coding-agent/src/extensions/codemode/tool.ts`` and
``execute.ts``. This module owns the agent-facing surface: the ``codemode``
AgentTool declaration and description (tool catalog grouped by MCP namespace
under a token budget), the session-backed store, and the rendering of a script
run into an AgentToolResult (output ordering, image files, token-budget
truncation, success/failure wrappers). Calls never touch the subprocess here:
execution goes through :func:`pi.codemode.host.execute_codemode`, and each
script call goes through ``execute_tool``, so nested calls run through the
agent pipeline (validation, hooks, permission checks, events, usage) and the
live call record is backfilled with the runner-assigned id.

Without an ``execute_tool`` callback the tool falls back to calling
``tool.execute`` directly. That path is for standalone use: it has no agent
hooks, no parent id, no events, and its usage is not aggregated.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..agent.types import AgentTool, AgentToolResult
from ..ai.types import ImageContent, TextContent
from ..coding_agent.nested_tool_calls import NestedToolCallOutcome
from ..coding_agent.session import SessionManager
from . import host

STORE_TYPE = "codemode-store"
MAX_OUTPUT_CHARS = 16 * 1024 * 1024
MAX_OUTPUT_ITEMS = 100_000
MAX_STORE_VALUE_CHARS = 256 * 1024
MAX_STORE_TOTAL_CHARS = 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_INLINE_BUDGET = 3_000
CHARS_PER_TOKEN = 4
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}


def load_codemode_store(session: Optional[SessionManager]) -> Dict[str, Any]:
    """Values of ``load()``: the ``codemode-store`` entries on the session, applied in order."""

    data: Dict[str, Any] = {}
    if session is None:
        return data
    for entry in session.custom_entries(STORE_TYPE):
        custom = entry.custom["data"] or {}
        for key in custom.get("delete") or []:
            data.pop(key, None)
        for key, value in (custom.get("set") or {}).items():
            data[key] = value
    return data


def _schema_type(schema: Any) -> str:
    if schema is True or schema == {} or schema is None:
        return "Any"
    if schema is False:
        return "Never"
    if not isinstance(schema, dict):
        return "Any"
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(_schema_type({"type": item}) for item in kind)
    if kind == "string":
        return "str"
    if kind == "integer":
        return "int"
    if kind == "number":
        return "float"
    if kind == "boolean":
        return "bool"
    if kind == "array":
        items = schema.get("items")
        return "list[" + _schema_type(items) + "]" if items is not None else "list"
    if kind == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            required = set(schema.get("required") or [])
            fields = []
            for key, sub in properties.items():
                if not isinstance(key, str):
                    continue
                rendered = f"{key}: {_schema_type(sub)}"
                fields.append(rendered if key in required else rendered + " = None")
            return "{" + ", ".join(fields) + "}"
        return "dict"
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        values = schema["enum"]
        head = ", ".join(repr(value) for value in values[:6])
        return head + (", ..." if len(values) > 6 else "")
    if isinstance(schema.get("anyOf"), list):
        return " | ".join(_schema_type(sub) for sub in schema["anyOf"])
    return "Any"


def _describe_output(output_schema: Optional[dict]) -> str:
    """What a script call resolves to: a string, the object's field names, or the
    rendered type for anything else such as a union."""

    if not isinstance(output_schema, dict):
        return "str"
    if isinstance(output_schema.get("anyOf"), list):
        return _schema_type(output_schema)
    if output_schema.get("type") != "object":
        return "str"
    properties = output_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "str"
    required = set(output_schema.get("required") or [])
    fields = [key if key in required else key + "?" for key in properties if isinstance(key, str)]
    return "{ " + ", ".join(fields) + " }"


def render_tool_sample(name: str, description: str, parameters: dict,
                       output_schema: Optional[dict] = None) -> str:
    """The declaration a script sees for one tool: signature plus description."""

    identifier = host.to_codemode_identifier(name)
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    required = set(parameters.get("required") or []) if isinstance(parameters, dict) else set()
    args_text = ""
    if isinstance(properties, dict) and properties:
        rendered = []
        for key, sub in properties.items():
            if not isinstance(key, str):
                continue
            arg = f"{key}: {_schema_type(sub)}"
            rendered.append(arg if key in required else arg + " = None")
        args_text = ", ".join(rendered)
    signature = f"tools.{identifier}({args_text}) -> {_describe_output(output_schema)}"
    lines = [signature]
    body = (description or "").strip()
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _select_catalog(groups: List[dict], budget: Optional[int]) -> set:
    """Pick tool sections that fit the budget: every group places its cheapest
    remaining section each round, so every namespace is represented first."""

    if budget is None:
        return {entry["name"] for group in groups for entry in group["entries"]}
    queues = [sorted(group["entries"], key=lambda entry: entry["cost"]) for group in groups]
    shown: set = set()
    remaining = budget
    active = [queue for queue in queues if queue]
    while active:
        surviving = []
        for queue in active:
            entry = queue[0]
            if entry["cost"] > remaining:
                continue
            remaining -= entry["cost"]
            shown.add(entry["name"])
            queue.pop(0)
            if queue:
                surviving.append(queue)
        active = surviving
    return shown


def _tool_namespace(name: str) -> Optional[str]:
    """Server namespace of an MCP tool name (``mcp__dev__tool`` -> ``dev``)."""

    match = re.match(r"mcp__([^_]+(?:_[^_]+)*)__", name)
    return match.group(1) if match else None


DESCRIPTION_INTRO = (
    "Run Python that calls other tools. The input is raw Python (not JSON, no code fence), "
    "executed in an isolated subprocess with file, import, and code-evaluation builtins "
    "disabled; tool calls are its only I/O.\n"
    "- tools.<name>(args) calls one tool and blocks until it finishes; tools.<name>.start(args) "
    "starts a call and returns a handle, composed with wait_all(...) / wait_any(...). "
    "Set the module variable `result` to return a value. Calls still running when the script ends are cancelled.\n"
    '- Optional first line: # @options: {"max_output_tokens": 10000, "timeout_ms": 60000}'
)

DESCRIPTION_GLOBALS = (
    "Globals:\n"
    "- text(value), image(dataUriOrImageBlock), print(...), and `result` add output; exit(value) ends the script.\n"
    "- store(key, value) and load(key) keep JSON values across codemode calls; store(key) deletes.\n"
    "- ALL_TOOLS lists every tool; search_tools(query, {limit?, namespace?}), "
    "describe_tool(name), describe_namespace(name) find unlisted tools, such as MCP tools."
)


def create_codemode_description(tools: List[AgentTool], inline_budget: Optional[int] = DEFAULT_INLINE_BUDGET) -> str:
    """Model-facing description: the helper list plus one section per listed tool,
    grouped by MCP namespace and limited to the token budget."""

    sections = [DESCRIPTION_INTRO, DESCRIPTION_GLOBALS]
    declarations = []
    for tool in tools:
        if tool.name == "codemode":
            continue
        sample = render_tool_sample(tool.name, tool.description or "",
                                    tool.parameters or {}, tool.output_schema)
        declarations.append({"name": tool.name, "sample": sample})
    if not declarations:
        return "\n\n".join(sections)

    groups: Dict[str, dict] = {"": {"namespace": None, "entries": []}}
    for declaration in declarations:
        namespace = _tool_namespace(declaration["name"])
        key = "ns:" + namespace if namespace else ""
        group = groups.setdefault(key, {"namespace": namespace, "entries": []})
        group["entries"].append({
            "name": declaration["name"], "sample": declaration["sample"],
            "cost": math.ceil(len(declaration["sample"]) / CHARS_PER_TOKEN),
        })
    ordered = sorted(groups.values(),
                     key=lambda group: (group["namespace"] is not None, group["namespace"] or ""))
    shown = _select_catalog(ordered, inline_budget)

    tool_sections = ["Nested tools:"]
    for group in ordered:
        namespace, entries = group["namespace"], group["entries"]
        visible = [entry for entry in entries if entry["name"] in shown]
        if namespace:
            listing = ""
            if len(visible) != len(entries):
                listing = " (tools not listed)" if not visible else " (some tools not listed)"
            tool_sections.append(f"## {namespace}{listing}")
        for entry in visible:
            identifier = host.to_codemode_identifier(entry["name"])
            heading = f"### `{identifier}`" if identifier == entry["name"] \
                else f"### `{identifier}` (`{entry['name']}`)"
            tool_sections.append(f"{heading}\n{entry['sample'].strip()}")
    sections.append("\n\n".join(tool_sections))
    return "\n\n".join(sections)


class _Truncation:
    """Token-budget truncation: keep head and tail of the combined text, spill the rest."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def apply(self, items: List[Union[TextContent, ImageContent]], max_tokens: int, call_id: str
              ) -> List[Union[TextContent, ImageContent]]:
        texts = [item.text for item in items if isinstance(item, TextContent)]
        combined = "\n".join(texts)
        budget = max_tokens * CHARS_PER_TOKEN
        if not texts or len(combined) <= budget:
            return items
        head = combined[:budget // 2]
        tail = combined[len(combined) - (budget - budget // 2):] if budget - budget // 2 > 0 else ""
        removed = len(combined) - budget
        warning = (f"Warning: truncated output (original token count: {math.ceil(len(combined) / CHARS_PER_TOKEN)})\n"
                   f"Total output lines: {combined.count(chr(10)) + 1}\n\n"
                   f"{head}...{math.ceil(removed / CHARS_PER_TOKEN)} tokens truncated...{tail}")
        spill = self.workdir / f"codemode_{host.to_codemode_file_stem(call_id)}_full_output.txt"
        try:
            spill.write_text(combined, encoding="utf-8")
            warning += f"\n\n[Full output: {spill} (read with offset/limit)]"
        except OSError as exc:
            warning += f"\n\n[Could not save the full output: {exc}]"
        return [TextContent(text=warning)] + [item for item in items if isinstance(item, ImageContent)]


def _save_images(items: List[Union[TextContent, ImageContent]], workdir: Path, call_id: str
                 ) -> List[Union[TextContent, ImageContent]]:
    """Save each output image to a temp file and put a text item with its path before it."""

    saved: Dict[str, str] = {}
    result: List[Union[TextContent, ImageContent]] = []
    counter = 0
    for item in items:
        if not isinstance(item, ImageContent):
            result.append(item)
            continue
        path = saved.get(item.data)
        if path is None:
            extension = IMAGE_EXTENSIONS.get(item.mime_type)
            if extension is None:
                result.append(TextContent(text=f"[Image ({item.mime_type}) has no known file extension]"))
                result.append(item)
                continue
            counter += 1
            path = str(workdir / f"codemode_{host.to_codemode_file_stem(call_id)}_img{counter}{extension}")
            try:
                Path(path).write_bytes(base64.b64decode(item.data))
            except OSError as exc:
                path = ""
                result.append(TextContent(text=f"[Image ({item.mime_type}) could not be saved: {exc}]"))
                result.append(item)
                continue
            saved[item.data] = path
        if path:
            size = len(base64.b64decode(item.data))
            result.append(TextContent(text=f"[Image saved to {path} ({item.mime_type}, {size} bytes)]"))
        result.append(item)
    return result


def _result_text(result: AgentToolResult) -> str:
    return "\n".join(block.text for block in (result.content or []) if isinstance(block, TextContent))


def _build_output_items(output_items: List[dict], extra_text: str) -> List[Union[TextContent, ImageContent]]:
    """Output items in production order: text first, images after, then the extra text."""

    items: List[Union[TextContent, ImageContent]] = []
    for item in output_items:
        if item.get("type") == "text":
            items.append(TextContent(text=str(item.get("text", ""))))
    if extra_text:
        items.append(TextContent(text=extra_text))
    for item in output_items:
        if item.get("type") == "image" and isinstance(item.get("data"), str):
            items.append(ImageContent(data=item["data"], mime_type=str(item.get("mime_type", "image/png"))))
    return items


def _failed_result(call_id: str, output_items: List[dict], calls: List[dict], error_text: str,
                   wall_time: float = 0.0) -> AgentToolResult:
    """Failure wrapper: header, output produced so far, then the error and the
    calls made before the failure (they are not undone)."""

    if calls:
        summary = "Tool calls made before the failure (they are not undone): " + ", ".join(
            f"{record['name']} ({record['status']})" for record in calls)
    else:
        summary = "No tool calls were made."
    header = f"Script failed\nWall time {wall_time:.1f} seconds\nOutput:"
    items = _build_output_items(output_items, f"{error_text}\n\n{summary}".strip())
    return AgentToolResult(content=[TextContent(text=header)] + items,
                           details={"calls": calls}, is_error=True)


def create_codemode_tool(tools: List[AgentTool], session: Optional[SessionManager] = None,
                         settings: Optional[dict] = None, execute_tool=None) -> AgentTool:
    """Build the ``codemode`` AgentTool over the given callable tools.

    ``execute_tool(name, args, signal)`` returns a
    :class:`~pi.coding_agent.nested_tool_calls.NestedToolCallOutcome` and runs
    the call through the agent pipeline, so nested calls get the same
    validation, hooks, permission checks, events, and usage accounting as
    direct calls. The CLI always passes the unified callback.
    """

    settings = settings or {}
    configured_timeout = settings.get("timeout_ms")
    inline_budget = settings.get("inline_budget", DEFAULT_INLINE_BUDGET)
    callable_tools = [tool for tool in tools if tool.name != "codemode"]
    allowed = {tool.name: tool for tool in callable_tools}
    runtime = host.prepare_runtime()

    js_names: Dict[str, str] = {}
    tool_meta: List[dict] = []
    for tool in callable_tools:
        js = host.to_codemode_identifier(tool.name)
        if js in js_names:
            js = ""
        else:
            js_names[js] = tool.name
        tool_meta.append({
            "name": tool.name, "js_name": js,
            "namespace": _tool_namespace(tool.name),
            "description": render_tool_sample(tool.name, tool.description or "",
                                              tool.parameters or {}, tool.output_schema),
        })

    async def run_tool(name: str, args: dict, signal,
                       fallback_id: str = "") -> NestedToolCallOutcome:
        if execute_tool is not None:
            return await execute_tool(name, args, signal)
        tool = allowed.get(name)
        if tool is None:
            raise ValueError(f"tool not available: {name}")
        call_id = fallback_id or f"codemode-{name}"
        result = await tool.execute(call_id, args, signal, None)
        return NestedToolCallOutcome(
            tool_call_id=call_id, result=result, is_error=bool(result.is_error)
        )

    async def run_script_tool(name: str, args: dict, signal,
                              record: host.CallRecord) -> Any:
        """Execute one script call and convert the full outcome for the script."""
        outcome = await run_tool(name, args, signal, record.id)
        # Show the id the runner actually assigned, not a local guess.
        record.id = outcome.tool_call_id
        if outcome.is_error:
            raise RuntimeError(_result_text(outcome.result) or f'tool "{name}" failed')
        structured = outcome.result.structured_content
        if structured is not None:
            return structured
        return _result_text(outcome.result)

    async def execute(call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("codemode requires a code string")
        try:
            options, code = host.parse_codemode_source(code)
        except ValueError as exc:
            return _failed_result(call_id, [], [], f"Script error:\n{exc}")
        timeout_ms = options.get("timeout_ms", configured_timeout)
        max_tokens = options.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)

        def publish(snapshot: dict) -> None:
            if on_update is not None:
                update = AgentToolResult(content=[], details=snapshot)
                outcome = on_update(update)
                if asyncio.iscoroutine(outcome):
                    asyncio.ensure_future(outcome)

        outcome = await host.execute_codemode(
            code=code, root_id=call_id, tools_meta=tool_meta,
            store=load_codemode_store(session), timeout_ms=timeout_ms,
            signal=signal, on_call=run_script_tool,
            on_update=publish if on_update is not None else None, runtime=runtime)

        if outcome.status == host.STATUS_ABORTED:
            return _failed_result(call_id, outcome.output_items, outcome.calls,
                                  "Script aborted: codemode script was aborted", outcome.wall_time)
        if outcome.status == host.STATUS_TIMEOUT:
            return _failed_result(call_id, outcome.output_items, outcome.calls,
                                  f"Script timed out: exceeded the {timeout_ms} ms deadline", outcome.wall_time)
        if outcome.status == host.STATUS_CRASHED:
            return _failed_result(call_id, outcome.output_items, outcome.calls,
                                  f"Script sandbox failed: codemode worker exited unexpectedly (exit code {outcome.exit_code})",
                                  outcome.wall_time)
        if outcome.status == host.STATUS_SCRIPT_ERROR:
            error = outcome.error or {}
            stack = error.get("stack") or f'{error.get("name", "Error")}: {error.get("message", "")}'
            return _failed_result(call_id, outcome.output_items, outcome.calls,
                                  "Script error:\n" + stack.strip(), outcome.wall_time)

        if outcome.writes["set"] or outcome.writes["delete"]:
            if session is not None:
                session.append_custom(STORE_TYPE, {"set": dict(outcome.writes["set"]),
                                                   "delete": list(outcome.writes["delete"])})

        value = outcome.value
        extra = "" if value is None else (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))
        items = _build_output_items(outcome.output_items, extra)
        items = _save_images(items, runtime.workdir, call_id)
        items = _Truncation(runtime.workdir).apply(items, max_tokens, call_id)
        return AgentToolResult(
            content=[TextContent(text=f"Script completed\nWall time {outcome.wall_time:.1f} seconds\nOutput:")] + items,
            details={"calls": outcome.calls})

    return AgentTool(
        name="codemode",
        description=create_codemode_description(callable_tools, inline_budget),
        parameters={"type": "object", "properties": {"code": {"type": "string", "description": "Raw Python source."}},
                    "required": ["code"]},
        execute=execute,
    )
