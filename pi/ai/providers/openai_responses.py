"""OpenAI Responses HTTP/SSE adapter."""

import json
import time
import os
from urllib.parse import urlencode

from ..types import UserMessage, AssistantMessage, ToolResultMessage, ToolCall, TextContent, ImageContent, AudioContent, VideoContent, ThinkingContent, get_system_message_text, get_message_text
from ..utils import normalize_context
from .base import SimpleStreamOptions, apply_model_params, run_provider_stream, usage_from_counts
from .blocks import BlockEvents
from .copilot import copilot_headers

# Output item types with a dedicated replay shape; anything else arriving on
# response.output_item.done is a provider-executed (hosted) tool call.
_NON_HOSTED_ITEM_TYPES = ("message", "reasoning", "function_call")

# Chat/Responses audio input accepts only these formats, keyed by MIME type.
_AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}


def _audio_part(block: AudioContent) -> dict:
    fmt = _AUDIO_FORMATS.get(block.mime_type.split(";")[0].strip().lower())
    if fmt is None:
        raise ValueError(f"Unsupported audio type {block.mime_type}: OpenAI accepts audio/wav and audio/mpeg")
    return {"type": "input_audio", "input_audio": {"data": block.data, "format": fmt}}


def build_request(model, context, options):
    system, messages = normalize_context(context)
    items = []
    for message in messages:
        if isinstance(message, UserMessage):
            blocks = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
            content = []
            for b in blocks:
                if isinstance(b, ImageContent):
                    content.append({"type": "input_image", "image_url": f"data:{b.mime_type};base64,{b.data}"})
                elif isinstance(b, AudioContent):
                    content.append(_audio_part(b))
                elif isinstance(b, VideoContent):
                    raise ValueError("OpenAI Responses does not support video input")
                else:
                    content.append({"type": "input_text", "text": b.text})
            items.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            same_model = (message.provider == model.provider and message.api == model.api
                          and message.model == model.id)
            for block in message.content:
                if isinstance(block, TextContent):
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": block.text}]})
                elif isinstance(block, ToolCall):
                    if block.raw_item is not None:
                        # Hosted (provider-executed) call: replay the raw output
                        # item verbatim for stateless chaining.
                        items.append(block.raw_item)
                    else:
                        items.append({"type": "function_call", "call_id": block.id, "name": block.name, "arguments": json.dumps(block.arguments)})
                elif isinstance(block, ThinkingContent):
                    if same_model and block.thinking_signature:
                        items.append(json.loads(block.thinking_signature))
                    elif not block.redacted and block.thinking.strip():
                        items.append({"role": "assistant", "content": [{"type": "output_text", "text": block.thinking}]})
        elif isinstance(message, ToolResultMessage):
            output = [{"type": "input_image", "image_url": f"data:{b.mime_type};base64,{b.data}"} if isinstance(b, ImageContent)
                      else {"type": "input_text", "text": b.text} for b in message.content]
            items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": output or "(empty result)"})
    body = {"model": model.id, "input": items, "stream": True, "store": False,
            "max_output_tokens": options.max_tokens or model.max_tokens, "include": ["reasoning.encrypted_content"]}
    if system:
        body["instructions"] = get_system_message_text(system)
        if system.tools_added:
            body["tools"] = [t.hosted if t.hosted else
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": False}
                for t in system.tools_added]
    if model.reasoning and options.reasoning and options.reasoning != "off":
        body["reasoning"] = {"effort": options.reasoning, "summary": "auto"}
    if options.temperature is not None:
        body["temperature"] = options.temperature
    if model.api == "openai-codex-responses":
        body.pop("max_output_tokens", None)
        body.setdefault("instructions", "You are a helpful assistant.")
        body["parallel_tool_calls"] = True
        body["text"] = {"verbosity": "low"}
        for tool in body.get("tools", []):
            if tool.get("type") == "function":
                tool["strict"] = None
    apply_model_params(body, model, options)
    if model.api == "azure-openai-responses":
        body.pop("api_version", None)
        body["model"] = model.params.get("deployment_name", model.id)
        body.pop("deployment_name", None)
        body["max_output_tokens"] = max(16, body["max_output_tokens"])
    return body


async def stream_openai_responses(model, context, options=None):
    options = options or SimpleStreamOptions()
    blocks = None
    async def convert(event_name, data, state):
        nonlocal blocks
        blocks = blocks or BlockEvents(state)
        kind = data.get("type", event_name) if isinstance(data, dict) else ""
        key = data.get("output_index", 0) if isinstance(data, dict) else 0
        events = []
        if kind in ("response.refusal.delta", "response.refusal.done"):
            raise RuntimeError(f"Provider refusal: {data.get('delta') or data.get('refusal') or 'request refused'}")
        if kind == "response.output_item.done" and data["item"].get("type") == "message":
            refusals = [part.get("refusal", "request refused") for part in data["item"].get("content", [])
                        if part.get("type") == "refusal"]
            if refusals:
                raise RuntimeError("Provider refusal: " + "\n".join(refusals))
        if kind == "response.output_text.delta":
            events = blocks.text((key, data.get("content_index", 0)), data["delta"])
        elif kind == "response.reasoning_summary_text.delta":
            events = blocks.text(("thinking", key), data["delta"], thinking=True)
        elif kind == "response.output_item.added" and data["item"]["type"] == "function_call":
            item = data["item"]
            events = blocks.tool(("tool", key), item["call_id"], item["name"], item.get("arguments", ""))
        elif kind == "response.function_call_arguments.delta":
            events = blocks.tool(("tool", key), "", "", data["delta"])
        elif kind == "response.function_call_arguments.done":
            tool_key = ("tool", key)
            blocks.arguments[tool_key] = data["arguments"]
            state.partial.content[blocks.indices[tool_key]].arguments = json.loads(data["arguments"])
        elif kind == "response.output_item.done" and data["item"]["type"] == "reasoning":
            item = data["item"]
            index = blocks.indices.get(("thinking", key))
            if index is None:
                events = blocks.text(("thinking", key), "", thinking=True)
                index = blocks.indices[("thinking", key)]
            state.partial.content[index].thinking_signature = json.dumps(item)
        elif kind == "response.output_item.done" and data["item"]["type"] not in _NON_HOSTED_ITEM_TYPES:
            events = blocks.hosted_call(("hosted", key), data["item"])
        elif kind in ("response.completed", "response.incomplete"):
            response = data["response"]
            refusals = [part.get("refusal", "request refused") for item in response.get("output", [])
                        if item.get("type") == "message" for part in item.get("content", [])
                        if part.get("type") == "refusal"]
            if refusals:
                raise RuntimeError("Provider refusal: " + "\n".join(refusals))
            usage = response.get("usage") or {}
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            state.set_usage(usage_from_counts(max(0, usage.get("input_tokens", 0) - cached), usage.get("output_tokens", 0), cached,
                reasoning=(usage.get("output_tokens_details") or {}).get("reasoning_tokens")))
            state.partial.usage.cost = model.calculate_cost(state.partial.usage)
            state.set_response(response.get("id"), response.get("model"))
            executed = any(isinstance(b, ToolCall) and b.raw_item is None for b in state.partial.content)
            status = response.get("status")
            state.set_raw_stop_reason(status or kind)
            if status not in (None, "completed", "incomplete"):
                raise RuntimeError(f"Provider response status: {status}")
            if kind == "response.incomplete":
                incomplete = (response.get("incomplete_details") or {}).get("reason")
                if incomplete != "max_output_tokens":
                    raise RuntimeError(f"Response incomplete: {incomplete or 'without a provider reason'}")
            reason = "length" if kind == "response.incomplete" else "toolUse" if executed else "stop"
            events = blocks.finish(reason)
        elif kind in ("error", "response.failed"):
            raise RuntimeError(str(data.get("error") or data.get("response", {}).get("error") or data.get("message", "Responses request failed")))
        for event in events:
            yield event
    path = "/codex/responses" if model.api == "openai-codex-responses" else "/responses"
    azure = model.api == "azure-openai-responses"
    if azure:
        path += "?" + urlencode({"api-version": model.params.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "v1"})
    def headers():
        key = options.api_key or model.api_key or ""
        return {**({"api-key": key} if azure and model.auth_type != "entra" else {"Authorization": "Bearer " + key}),
                "content-type": "application/json",
                **({"originator": "pi", "OpenAI-Beta": "responses=experimental"} if model.api == "openai-codex-responses" else {}),
                **model.headers, **copilot_headers(model, context)}
    async for event in run_provider_stream(model, lambda: build_request(model, context, options), path,
        headers, convert, options, int(time.time() * 1000)):
        yield event
