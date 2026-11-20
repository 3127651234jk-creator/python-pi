"""Gemini generateContent HTTP/SSE adapter."""

import json
import re
import time
from urllib.parse import quote

from ..types import TextContent, ImageContent, AudioContent, VideoContent, ThinkingContent, ToolCall, ToolResultMessage, get_system_message_text
from ..utils import normalize_context
from .base import SimpleStreamOptions, run_provider_stream, usage_from_counts, thinking_budget, apply_model_params, resolve_max_tokens
from .blocks import BlockEvents


def _gemini_major_version(model_id: str):
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    return int(match.group(1)) if match else None


def requires_tool_call_id(model_id: str) -> bool:
    """Models that require explicit ids on function calls and responses (TS
    ``requiresToolCallId``): Gemini 3+, Claude and gpt-oss behind Google APIs."""
    major = _gemini_major_version(model_id)
    return (model_id.startswith("claude-") or model_id.startswith("gpt-oss-")
            or (major is not None and major >= 3))


def _supports_multimodal_function_response(model_id: str) -> bool:
    """Gemini 3+ accepts images nested in functionResponse.parts; older Gemini
    needs a separate user image turn, and non-Gemini models accept nested."""
    major = _gemini_major_version(model_id)
    return True if major is None else major >= 3


def build_request(model, context, options):
    system, messages = normalize_context(context)
    include_ids = requires_tool_call_id(model.id)
    supports_images = "image" in model.input
    multimodal_tool_images = _supports_multimodal_function_response(model.id)
    contents = []
    for message in messages:
        same_model = (message.role == "assistant" and message.provider == model.provider
                      and message.api == model.api and message.model == model.id)
        parts = []
        if isinstance(message, ToolResultMessage):
            text = "\n".join(b.text for b in message.content if isinstance(b, TextContent))
            images = [b for b in message.content if isinstance(b, ImageContent)] if supports_images else []
            image_parts = [{"inlineData": {"mimeType": b.mime_type, "data": b.data}} for b in images]
            function_response = {
                "name": message.tool_name,
                "response": {"error" if message.is_error else "output":
                             text if text else "(see attached image)" if images else ""},
            }
            if images and multimodal_tool_images:
                function_response["parts"] = image_parts
            if include_ids:
                function_response["id"] = message.tool_call_id
            part = {"functionResponse": function_response}
            last = contents[-1] if contents else None
            if last is not None and last["role"] == "user" and any(
                    "functionResponse" in item for item in last["parts"]):
                last["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            if images and not multimodal_tool_images:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})
            continue
        for block in [TextContent(text=message.content)] if isinstance(message.content, str) else message.content:
            if isinstance(block, TextContent):
                signature = block.text_signature if same_model else None
                # Empty text is dropped unless it carries a signature Gemini
                # requires echoed back (TS keeps signature-only parts).
                if message.role == "assistant" and not block.text.strip() and not signature:
                    continue
                part = {"text": block.text}
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            elif isinstance(block, (ImageContent, AudioContent, VideoContent)):
                parts.append({"inlineData": {"mimeType": block.mime_type, "data": block.data}})
            elif isinstance(block, ToolCall):
                if block.raw_item is not None:
                    continue  # Hosted calls are provider-executed; not replayable on Gemini.
                function_call = {"name": block.name, "args": block.arguments}
                if include_ids:
                    function_call["id"] = block.id
                part = {"functionCall": function_call}
                if same_model and block.thought_signature:
                    part["thoughtSignature"] = block.thought_signature
                parts.append(part)
            elif isinstance(block, ThinkingContent):
                if same_model and block.thinking_signature:
                    parts.append({"text": block.thinking, "thought": True, "thoughtSignature": block.thinking_signature})
                elif not block.redacted and block.thinking.strip():
                    parts.append({"text": block.thinking})
        role = "model" if message.role == "assistant" else "user"
        if parts:
            if contents and contents[-1]["role"] == role and not any(
                    "functionResponse" in item for item in contents[-1]["parts"]):
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
    body = {"contents": contents, "generationConfig": {"maxOutputTokens": resolve_max_tokens(model, context, options)}}
    if options.temperature is not None:
        body["generationConfig"]["temperature"] = options.temperature
    if model.reasoning and options.reasoning and options.reasoning != "off":
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget(options.reasoning), "includeThoughts": True}
    if system:
        body["systemInstruction"] = {"parts": [{"text": get_system_message_text(system)}]}
        functions = [{"name": t.name, "description": t.description, "parametersJsonSchema": t.parameters}
            for t in (system.tools_added or []) if not t.hosted]
        if functions:
            body["tools"] = [{"functionDeclarations": functions}]
    # Model-specific params tune generationConfig (top_k, temperature, ...).
    explicit = set()
    if options.temperature is not None:
        explicit.add("temperature")
    if options.max_tokens is not None:
        explicit.add("maxOutputTokens")
    if options.reasoning is not None:
        explicit.add("thinkingConfig")
    config_params = {k: v for k, v in (model.params or {}).items() if k not in explicit}
    body["generationConfig"].update(config_params)
    return body


async def stream_google(model, context, options=None):
    options = options or SimpleStreamOptions()
    convert = make_google_converter(model)
    path = "/models/" + quote(model.id.removeprefix("models/"), safe="") + ":streamGenerateContent?alt=sse"
    async for event in run_provider_stream(model, lambda: build_request(model, context, options), path,
        lambda: {"x-goog-api-key": options.api_key or model.api_key or "", "content-type": "application/json", **model.headers}, convert, options, int(time.time() * 1000)):
        yield event


def make_google_converter(model):
    """SSE-to-pi-event converter for generateContent responses (Google AI + Vertex)."""
    blocks = None
    tool_call_counter = 0
    segment = 0
    current_key = None
    current_kind = None

    async def convert(event_name, data, state):
        nonlocal blocks, tool_call_counter, segment, current_key, current_kind
        blocks = blocks or BlockEvents(state)
        if not isinstance(data, dict):
            return
        if data.get("error") or data.get("promptFeedback", {}).get("blockReason"):
            raise RuntimeError(str(data.get("error") or data["promptFeedback"]))
        usage = data.get("usageMetadata", {})
        if usage:
            cached = usage.get("cachedContentTokenCount", 0)
            thoughts = usage.get("thoughtsTokenCount", 0)
            state.set_usage(usage_from_counts(max(0, usage.get("promptTokenCount", 0) - cached),
                usage.get("candidatesTokenCount", 0) + thoughts, cached, reasoning=thoughts))
            state.partial.usage.cost = model.calculate_cost(state.partial.usage)
        state.set_response(data.get("responseId"), data.get("modelVersion"))
        for candidate in data.get("candidates", [])[:1]:
            for part in candidate.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    # A tool call ends any streamed text block; later text starts
                    # a fresh block so content order matches the wire parts.
                    current_key = None
                    current_kind = None
                    call = part["functionCall"]
                    provided_id = call.get("id")
                    existing_ids = {block.id for block in state.partial.content if isinstance(block, ToolCall)}
                    if not provided_id or provided_id in existing_ids:
                        tool_call_counter += 1
                        call_id = f"{call.get('name', '')}_{int(time.time() * 1000)}_{tool_call_counter}"
                    else:
                        call_id = provided_id
                    key = ("tool", len(blocks.arguments))
                    events = blocks.tool(key, call_id, call.get("name", ""),
                                         json.dumps(call.get("args", {})))
                    if part.get("thoughtSignature"):
                        state.partial.content[blocks.indices[key]].thought_signature = part["thoughtSignature"]
                elif "text" in part:
                    kind = "thinking" if part.get("thought") else "text"
                    if current_kind != kind:
                        segment += 1
                        current_kind = kind
                        current_key = (kind, segment)
                    events = blocks.text(current_key, part["text"], thinking=(kind == "thinking"))
                    signature = part.get("thoughtSignature")
                    if signature:
                        block = state.partial.content[blocks.indices[current_key]]
                        if kind == "thinking":
                            block.thinking_signature = signature
                        else:
                            block.text_signature = signature
                else:
                    events = []
                for event in events:
                    yield event
            finish = candidate.get("finishReason")
            if finish:
                state.set_raw_stop_reason(finish)
                if finish not in ("STOP", "MAX_TOKENS"):
                    raise RuntimeError(f"Gemini stopped: {finish}")
                reason = "length" if finish == "MAX_TOKENS" else "toolUse" if blocks.arguments else "stop"
                for event in blocks.finish(reason):
                    yield event
    return convert
