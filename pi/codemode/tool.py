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
                       output_schema: Optional[dict] = None,
                       guidelines: Optional[List[str]] = None) -> str:
    """The declaration a script sees for one tool: signature, description and guidelines."""

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
    bullets = [guideline.strip() for guideline in (guidelines or []) if guideline.strip()]
    if bullets:
        body = (body + "\n\n" if body else "") + "\n".join(f"- {bullet}" for bullet in bullets)
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
    "- text(value), image(dataUriOrImageBlock), print(...), and `result` add output; exit(value) ends the script. "
    "With several text items, each starts with a `==> text N/M <==` line, and console lines follow the other "
    "output in one `<console_output>` block.\n"
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
        if tool.name == "codemode" or tool.exposure == "model-only":
            continue
        sample = render_tool_sample(tool.name, tool.description or "",
                                    tool.parameters or {}, tool.output_schema,
                                    tool.prompt_guidelines)
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
    """Output items in production order: text first, images after, then the extra text.

    Providers join adjacent text blocks with a newline or with nothing, so with more than one
    text item (text() or the returned value) each starts with a ``==> text N/M <==`` line;
    console lines follow all other output in one ``<console_output>`` block.
    """

    texts: List[str] = []
    console_lines: List[str] = []
    for item in output_items:
        if item.get("type") != "text":
            continue
        if item.get("console"):
            console_lines.append(str(item.get("text", "")))
        else:
            texts.append(str(item.get("text", "")))
    if extra_text:
        texts.append(extra_text)
    items: List[Union[TextContent, ImageContent]] = []
    if len(texts) > 1:
        items.extend(TextContent(text=f"==> text {index}/{len(texts)} <==\n{text}")
                     for index, text in enumerate(texts, 1))
    elif texts:
        items.append(TextContent(text=texts[0]))
    for item in output_items:
        if item.get("type") == "image" and isinstance(item.get("data"), str):
            items.append(ImageContent(data=item["data"], mime_type=str(item.get("mime_type", "image/png"))))
    if console_lines:
        items.append(TextContent(text="<console_output>\n" + "\n".join(console_lines) + "\n</console_output>"))
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
    # The error wrapper is not script output: it stays unnumbered and follows the
    # console block, like the TS host pushing the error after formatOutput.
    items = _build_output_items(output_items, "")
    items.append(TextContent(text=f"{error_text}\n\n{summary}".strip()))
    return AgentToolResult(content=[TextContent(text=header)] + items,
                           details={"calls": calls}, is_error=True)


def create_codemode_tool(tools: List[AgentTool], session: Optional[SessionManager] = None,
                         settings: Optional[dict] = None, execute_tool=None,
                         model_registry=None) -> AgentTool:
    """Build the ``codemode`` AgentTool over the given callable tools.

    ``execute_tool(name, args, signal)`` returns a
    :class:`~pi.coding_agent.nested_tool_calls.NestedToolCallOutcome` and runs
    the call through the agent pipeline, so nested calls get the same
    validation, hooks, permission checks, events, and usage accounting as
    direct calls. The CLI always passes the unified callback.
    ``model_registry`` powers the script's ``models.*`` namespace (model
    lookup and classifiers); without one those calls fail.
    """

    state = {"settings": settings or {}}
    runtime = host.prepare_runtime()

    def current_tools() -> List[AgentTool]:
        # The list is owned by the agent and changes on MCP/extension refreshes;
        # read it at call time so codemode sees tools registered after setup.
        return [tool for tool in tools if tool.name != "codemode" and tool.exposure != "model-only"]

    def build_tool_meta() -> List[dict]:
        js_names: Dict[str, str] = {}
        meta: List[dict] = []
        for tool in current_tools():
            js = host.to_codemode_identifier(tool.name)
            if js in js_names:
                js = ""
            else:
                js_names[js] = tool.name
            meta.append({
                "name": tool.name, "js_name": js,
                "namespace": _tool_namespace(tool.name),
                "description": render_tool_sample(tool.name, tool.description or "",
                                                  tool.parameters or {}, tool.output_schema,
                                                  tool.prompt_guidelines),
            })
        return meta

    async def run_tool(name: str, args: dict, signal,
                       fallback_id: str = "") -> NestedToolCallOutcome:
        if execute_tool is not None:
            return await execute_tool(name, args, signal)
        tool = next((candidate for candidate in current_tools() if candidate.name == name), None)
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

    # -- models.* namespace (F-CA-04) ---------------------------------------

    MODEL_TYPES = ("chat", "image", "classifier")
    CLASSIFIER_CONTEXT_SHAPE = (
        '{ state: { ... }, images?: [{ type: "image", data: <base64>, mimeType }], '
        'questions: { <id>: { type: "choice", instructions, criteria: { <label>: <meaning> } } | '
        '{ type: "score", instructions, criteria: [<lowest level>, ..., <highest level>] } | '
        '{ type: "bool", instructions, criteria: { true: <meaning>, false: <meaning> } } } }')

    def _model_info(model):
        from ..coding_agent.rpc_wire import model_to_ts
        info = model_to_ts(model)
        info.pop("headers", None)  # models.json headers can carry credentials
        return info

    def _fail_shape(problem: str) -> ValueError:
        return ValueError(
            f"models.classify() {problem}. Expected context: {CLASSIFIER_CONTEXT_SHAPE}. "
            'See "Classify" in the codemode docs.')

    def check_classifier_context(context: Any) -> dict:
        """Validate a script's classifier context so mistakes fail with the
        expected shape instead of a provider error (port of checkClassifierContext)."""

        def is_record(value: Any) -> bool:
            return isinstance(value, dict)

        def describe_value(value: Any) -> str:
            if value is None:
                return "None"
            if isinstance(value, list):
                return "an empty array" if not value else "an array"
            if isinstance(value, dict):
                keys = list(value)
                if not keys:
                    return "{}"
                return "{ " + ", ".join(keys[:6]) + (", ..." if len(keys) > 6 else "") + " }"
            if isinstance(value, str):
                return "a string"
            return f"a {type(value).__name__}"

        if not is_record(context):
            raise _fail_shape(f"expects a context object as its second argument, got {describe_value(context)}")
        if not is_record(context.get("state")):
            raise _fail_shape(f"context.state must be an object, got {describe_value(context.get('state'))}")
        images = context.get("images")
        if images is not None:
            if not isinstance(images, list):
                raise _fail_shape(f"context.images must be an array, got {describe_value(images)}")
            for index, image in enumerate(images):
                if (not isinstance(image, dict) or image.get("type") != "image"
                        or not isinstance(image.get("data"), str)
                        or not isinstance(image.get("mimeType"), str)):
                    raise _fail_shape(f"context.images[{index}] must be an image block, got {describe_value(image)}")
        questions = context.get("questions")
        if not is_record(questions) or not questions:
            raise _fail_shape(f"context.questions must map question IDs to questions, got {describe_value(questions)}")
        for question_id, question in questions.items():
            at = f"context.questions.{question_id}"
            if not is_record(question):
                raise _fail_shape(f"{at} must be a question object, got {describe_value(question)}")
            if not isinstance(question.get("instructions"), str):
                raise _fail_shape(f"{at}.instructions must be a string")
            criteria = question.get("criteria")
            if question.get("type") == "choice":
                if (not is_record(criteria) or not criteria
                        or not all(isinstance(value, str) for value in criteria.values())):
                    raise _fail_shape(f'{at} is a "choice" question, so criteria must map each label to its meaning')
            elif question.get("type") == "score":
                if (not isinstance(criteria, list) or not criteria
                        or not all(isinstance(value, str) for value in criteria)):
                    raise _fail_shape(f'{at} is a "score" question, so criteria must list the levels as strings, lowest first')
            elif question.get("type") == "bool":
                if (not is_record(criteria) or not isinstance(criteria.get("true"), str)
                        or not isinstance(criteria.get("false"), str)):
                    raise _fail_shape(f'{at} is a "bool" question, so criteria must be {{ true: string, false: string }}')
            else:
                raise _fail_shape(f'{at}.type must be "choice", "score", or "bool", got {json.dumps(question.get("type"))}')
        return context

    def _classifier_questions(questions: dict) -> Dict[str, Any]:
        from ..ai.types import (
            ClassifierBoolQuestion,
            ClassifierChoiceQuestion,
            ClassifierScoreQuestion,
        )
        built: Dict[str, Any] = {}
        for question_id, question in questions.items():
            if question["type"] == "choice":
                built[question_id] = ClassifierChoiceQuestion(
                    instructions=question["instructions"], criteria=dict(question["criteria"]))
            elif question["type"] == "score":
                built[question_id] = ClassifierScoreQuestion(
                    instructions=question["instructions"], criteria=list(question["criteria"]))
            else:
                built[question_id] = ClassifierBoolQuestion(
                    instructions=question["instructions"],
                    criteria={"true": question["criteria"]["true"], "false": question["criteria"]["false"]})
        return built

    def _classifier_result_to_script(result) -> dict:
        from dataclasses import asdict
        from ..ai.types import ClassifierResult as _Result  # noqa: F401 - shape reference
        data = {
            "api": result.api, "provider": result.provider, "model": result.model,
            "answers": {name: asdict(answer) for name, answer in result.answers.items()},
            "usage": None, "stopReason": result.stop_reason, "errorMessage": result.error_message,
            "timestamp": result.timestamp,
        }
        if result.usage is not None:
            data["usage"] = result.usage.to_dict()
        return data

    async def run_model_call(name: str, args: list, signal, record: host.CallRecord) -> Any:
        """Execute one models.* call: registry lookup and classifier runs in the
        host; a script-supplied baseUrl or headers never receives credentials."""
        if model_registry is None:
            raise ValueError("the models namespace is not available (no model registry)")
        if name in ("models.getModelsOfType", "models.getAvailableOfType"):
            if not isinstance(args, list) or len(args) != 2:
                raise ValueError(f"{name}() expects (type, provider=None)")
            model_type, provider = args
            if model_type not in MODEL_TYPES:
                raise ValueError(f'Unknown model type {json.dumps(model_type)}. Use "chat", "image", or "classifier".')
            if provider is not None and not isinstance(provider, str):
                raise ValueError("provider must be a string")
            return [_model_info(model) for model in model_registry.get_models_of_type(model_type, provider)]
        if name == "models.getModelOfType":
            if not isinstance(args, list) or len(args) != 3 or not all(isinstance(a, str) for a in args):
                raise ValueError('models.getModelOfType(type, provider, id) expects three strings. '
                                 'For example models.getModelOfType("classifier", "openai", "gpt-6-luna").')
            model_type, provider, model_id = args
            if model_type not in MODEL_TYPES:
                raise ValueError(f'Unknown model type {json.dumps(model_type)}. Use "chat", "image", or "classifier".')
            model = model_registry.get_model_of_type(model_type, provider, model_id)
            return _model_info(model) if model is not None else None
        if name == "models.classify":
            if not isinstance(args, list) or len(args) != 2:
                raise ValueError("models.classify() expects (model, context)")
            model_ref, context = args
            list_hint = 'List the classifier models you can use with models.getAvailableOfType("classifier").'
            if (not isinstance(model_ref, dict) or not isinstance(model_ref.get("provider"), str)
                    or not isinstance(model_ref.get("id"), str)):
                hint = (" models.getModelOfType() returns None for an unknown provider or id."
                        if model_ref is None else "")
                raise ValueError(f'models.classify() expects a classifier model as its first argument, '
                                 f'got {json.dumps(model_ref) if model_ref is not None else "None"}.{hint} {list_hint}')
            ref = f'{model_ref["provider"]}/{model_ref["id"]}'
            resolved = model_registry.get_model_of_type("classifier", model_ref["provider"], model_ref["id"])
            if resolved is None:
                raise ValueError(f'Unknown classifier model "{ref}". {list_hint}')
            checked = check_classifier_context(context)
            from ..ai.providers import SimpleStreamOptions, classify
            from ..ai.types import ClassifierContext
            images = [ImageContent(data=image["data"], mime_type=image["mimeType"])
                      for image in (checked.get("images") or [])]
            classifier_context = ClassifierContext(state=dict(checked["state"]),
                                                   questions=_classifier_questions(checked["questions"]),
                                                   images=images or None)
            result = await classify(resolved, classifier_context,
                                    SimpleStreamOptions(signal=signal))
            if result.usage is not None:
                record.cost = result.usage.cost.get("total", 0.0)
            return _classifier_result_to_script(result)
        raise ValueError(f"Unknown models namespace call: {name}")
        return _result_text(outcome.result)

    async def execute(call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("codemode requires a code string")
        try:
            options, code = host.parse_codemode_source(code)
        except ValueError as exc:
            return _failed_result(call_id, [], [], f"Script error:\n{exc}")
        timeout_ms = options.get("timeout_ms", state["settings"].get("timeout_ms"))
        max_tokens = options.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)

        def publish(snapshot: dict) -> None:
            if on_update is not None:
                update = AgentToolResult(content=[], details=snapshot)
                outcome = on_update(update)
                if asyncio.iscoroutine(outcome):
                    asyncio.ensure_future(outcome)

        outcome = await host.execute_codemode(
            code=code, root_id=call_id, tools_meta=build_tool_meta(),
            store=load_codemode_store(session), timeout_ms=timeout_ms,
            signal=signal, on_call=run_script_tool,
            on_model_call=run_model_call,
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

    codemode = AgentTool(
        name="codemode",
        description=create_codemode_description(
            current_tools(), state["settings"].get("inline_budget", DEFAULT_INLINE_BUDGET)),
        parameters={"type": "object", "properties": {"code": {"type": "string", "description": "Raw Python source."}},
                    "required": ["code"]},
        execute=execute,
    )

    def refresh_description() -> None:
        codemode.description = create_codemode_description(
            current_tools(), state["settings"].get("inline_budget", DEFAULT_INLINE_BUDGET))

    def update_settings(settings: dict) -> None:
        state["settings"] = settings or {}
        refresh_description()

    # Called by the CLI when tools, guidelines or codemode settings change.
    codemode._codemode_refresh = refresh_description
    codemode._codemode_update_settings = update_settings
    return codemode
