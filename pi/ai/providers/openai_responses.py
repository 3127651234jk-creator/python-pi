"""OpenAI Responses HTTP/SSE adapter."""

import json
import time

from ..types import UserMessage, AssistantMessage, ToolResultMessage, ToolCall, TextContent, ImageContent, ThinkingContent, get_system_message_text, get_message_text
from ..utils import normalize_context
from .base import SimpleStreamOptions, run_provider_stream, usage_from_counts
from .blocks import BlockEvents
from .copilot import copilot_headers


def build_request(model, context, options):
    system, messages = normalize_context(context)
    items = []
    for message in messages:
        if isinstance(message, UserMessage):
            blocks = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
            items.append({"role": "user", "content": [
                {"type": "input_image", "image_url": f"data:{b.mime_type};base64,{b.data}"} if isinstance(b, ImageContent)
                else {"type": "input_text", "text": b.text} for b in blocks]})
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextContent):
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": block.text}]})
                elif isinstance(block, ToolCall):
                    items.append({"type": "function_call", "call_id": block.id, "name": block.name, "arguments": json.dumps(block.arguments)})
                elif isinstance(block, ThinkingContent) and block.thinking_signature and message.api in ("openai-responses", "openai-codex-responses"):
                    items.append(json.loads(block.thinking_signature))
        elif isinstance(message, ToolResultMessage):
            output = [{"type": "input_image", "image_url": f"data:{b.mime_type};base64,{b.data}"} if isinstance(b, ImageContent)
                      else {"type": "input_text", "text": b.text} for b in message.content]
            items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": output or "(empty result)"})
    body = {"model": model.id, "input": items, "stream": True, "store": False,
            "max_output_tokens": options.max_tokens or model.max_tokens, "include": ["reasoning.encrypted_content"]}
    if system:
        body["instructions"] = get_system_message_text(system)
        if system.tools_added:
            body["tools"] = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": False} for t in system.tools_added]
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
            tool["strict"] = None
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
        if kind in ("response.output_text.delta", "response.refusal.delta"):
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
        elif kind in ("response.completed", "response.incomplete"):
            response = data["response"]
            usage = response.get("usage") or {}
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            state.set_usage(usage_from_counts(max(0, usage.get("input_tokens", 0) - cached), usage.get("output_tokens", 0), cached,
                reasoning=(usage.get("output_tokens_details") or {}).get("reasoning_tokens")))
            state.partial.usage.cost = model.calculate_cost(state.partial.usage)
            state.set_response(response.get("id"), response.get("model"))
            reason = "length" if kind == "response.incomplete" else "toolUse" if any(isinstance(b, ToolCall) for b in state.partial.content) else "stop"
            events = blocks.finish(reason)
        elif kind in ("error", "response.failed"):
            raise RuntimeError(str(data.get("error") or data.get("response", {}).get("error") or data.get("message", "Responses request failed")))
        for event in events:
            yield event
    path = "/codex/responses" if model.api == "openai-codex-responses" else "/responses"
    async for event in run_provider_stream(model, lambda: build_request(model, context, options), path,
        lambda: {"Authorization": "Bearer " + (options.api_key or model.api_key or ""), "content-type": "application/json",
            **({"originator": "pi", "OpenAI-Beta": "responses=experimental"} if model.api == "openai-codex-responses" else {}), **model.headers,
            **copilot_headers(model, context)}, convert, options, int(time.time() * 1000)):
        yield event
