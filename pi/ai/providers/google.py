"""Gemini generateContent HTTP/SSE adapter."""

import json
import time
import uuid
from urllib.parse import quote

from ..types import TextContent, ImageContent, ThinkingContent, ToolCall, ToolResultMessage, get_system_message_text
from ..utils import normalize_context
from .base import SimpleStreamOptions, run_provider_stream, usage_from_counts, thinking_budget, apply_model_params
from .blocks import BlockEvents


def build_request(model, context, options):
    system, messages = normalize_context(context)
    contents = []
    for message in messages:
        parts = []
        if isinstance(message, ToolResultMessage):
            parts = [{"functionResponse": {"name": message.tool_name, "id": message.tool_call_id,
                "response": {"error" if message.is_error else "output": "\n".join(b.text for b in message.content if isinstance(b, TextContent))}}}]
        else:
            for block in [TextContent(text=message.content)] if isinstance(message.content, str) else message.content:
                if isinstance(block, TextContent):
                    parts.append({"text": block.text})
                elif isinstance(block, ImageContent):
                    parts.append({"inlineData": {"mimeType": block.mime_type, "data": block.data}})
                elif isinstance(block, ToolCall):
                    part = {"functionCall": {"id": block.id, "name": block.name, "args": block.arguments}}
                    if block.thought_signature:
                        part["thoughtSignature"] = block.thought_signature
                    parts.append(part)
                elif isinstance(block, ThinkingContent) and block.thinking_signature:
                    parts.append({"text": block.thinking, "thought": True, "thoughtSignature": block.thinking_signature})
        role = "model" if message.role == "assistant" else "user"
        if parts:
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
    body = {"contents": contents, "generationConfig": {"maxOutputTokens": options.max_tokens or model.max_tokens}}
    if options.temperature is not None:
        body["generationConfig"]["temperature"] = options.temperature
    if model.reasoning and options.reasoning and options.reasoning != "off":
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget(options.reasoning), "includeThoughts": True}
    if system:
        body["systemInstruction"] = {"parts": [{"text": get_system_message_text(system)}]}
        if system.tools_added:
            body["tools"] = [{"functionDeclarations": [{"name": t.name, "description": t.description, "parametersJsonSchema": t.parameters} for t in system.tools_added]}]
    # Model-specific params tune generationConfig (top_k, temperature, ...).
    config_params = {k: v for k, v in (model.params or {}).items() if k not in ("temperature",) or options.temperature is None}
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

    async def convert(event_name, data, state):
        nonlocal blocks
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
                    call = part["functionCall"]
                    key = ("tool", len(blocks.arguments))
                    events = blocks.tool(key, call.get("id") or "google_" + uuid.uuid4().hex, call["name"], json.dumps(call.get("args", {})))
                    state.partial.content[blocks.indices[key]].thought_signature = part.get("thoughtSignature")
                elif "text" in part:
                    key = "thinking" if part.get("thought") else "text"
                    events = blocks.text(key, part["text"], thinking=bool(part.get("thought")))
                    if part.get("thoughtSignature") and part.get("thought"):
                        state.partial.content[blocks.indices[key]].thinking_signature = part["thoughtSignature"]
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
