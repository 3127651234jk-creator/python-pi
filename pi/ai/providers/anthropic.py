"""anthropic-messages provider adapter (streaming SSE).

Python port of ``packages/ai/src/api/anthropic-messages.ts``: request building
from a normalized transcript and SSE event mapping to the pi event protocol.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import Model
from ..types import (
    AssistantMessage,
    DoneEvent,
    ImageContent,
    StreamErrorEvent,
    SystemMessage,
    ThinkingLevel,
    ThinkingContent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    TextContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
    get_system_message_text,
)
from ..utils import normalize_context
from .copilot import copilot_headers
from .base import (
    ProviderState,
    SimpleStreamOptions,
    apply_model_params,
    run_provider_stream,
    thinking_budget,
    usage_from_counts,
)

_ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "toolUse",
    "max_tokens": "length",
    "refusal": "stop",
}


def build_request(model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions]) -> Dict[str, Any]:
    """Build the ``/v1/messages`` payload from a normalized transcript."""
    system, messages = normalize_context(context)
    cache_control = _cache_control(options.cache_retention if options else None)
    body: Dict[str, Any] = {
        "model": model.id,
        "max_tokens": (options.max_tokens if options and options.max_tokens else model.max_tokens),
        "stream": True,
        "messages": convert_messages(messages),
    }
    if system is not None:
        text = get_system_message_text(system)
        if text:
            if cache_control:
                body["system"] = [{"type": "text", "text": text, "cache_control": cache_control}]
            else:
                body["system"] = text
    tools = _declared_tools(system)
    if tools:
        body["tools"] = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in tools
        ]
        if cache_control:
            body["tools"][-1]["cache_control"] = cache_control
    if cache_control:
        _tag_last_message(body["messages"], cache_control)

    reasoning: Optional[ThinkingLevel] = options.reasoning if options else None
    budget = thinking_budget(reasoning) if model.reasoning else None
    if budget is not None:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["max_tokens"] = max(body["max_tokens"], budget + 4096)

    if options and options.temperature is not None and budget is None:
        body["temperature"] = options.temperature
    apply_model_params(body, model, options)
    return body


def _cache_control(retention: Optional[str]) -> Optional[Dict[str, str]]:
    """Anthropic cache marker for the requested retention; None disables caching."""
    if retention not in ("5m", "1h"):
        return None
    control = {"type": "ephemeral"}
    if retention == "1h":
        control["ttl"] = "1h"
    return control


def _tag_last_message(api_messages: List[Dict[str, Any]], cache_control: Dict[str, str]) -> None:
    """Attach the cache marker to the last block of the final message."""
    if not api_messages:
        return
    content = api_messages[-1]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
        api_messages[-1]["content"] = content
    if content:
        content[-1] = {**content[-1], "cache_control": cache_control}


def _declared_tools(system: Optional[SystemMessage]) -> List[Tool]:
    if system is None or not system.tools_added:
        return []
    return list(system.tools_added)


def convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert transcript messages to Anthropic API messages.

    Consecutive toolResult messages are merged into one user message holding
    multiple tool_result blocks, as the Anthropic API requires.
    """
    api_messages: List[Dict[str, Any]] = []
    pending_results: List[Dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            api_messages.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if isinstance(message, UserMessage):
            flush_results()
            api_messages.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            flush_results()
            api_messages.append({"role": "assistant", "content": _assistant_content(message)})
        elif isinstance(message, ToolResultMessage):
            content: List[Dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, ImageContent):
                    content.append(_image_block(block))
                else:
                    content.append({"type": "text", "text": block.text})
            if not content:
                content = [{"type": "text", "text": "(empty result)"}]
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": content,
                    "is_error": message.is_error,
                }
            )
        # SystemMessage: already folded by normalize_context
    flush_results()
    return api_messages


def _user_content(message: UserMessage) -> Any:
    if isinstance(message.content, str):
        return message.content
    blocks: List[Dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ImageContent):
            blocks.append(_image_block(block))
        else:
            blocks.append({"type": "text", "text": block.text})
    return blocks


def _image_block(block: ImageContent) -> Dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
    }


def _assistant_content(message: AssistantMessage) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            # Replay requires the provider signature; drop unsigned thinking.
            if block.thinking_signature:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.thinking_signature,
                    }
                )
        elif isinstance(block, ToolCall):
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments})
    if not blocks:
        blocks = [{"type": "text", "text": "(empty)"}]
    return blocks


async def _convert_events(event_name: Optional[str], data: Any, state: ProviderState) -> AsyncIterator[Any]:
    partial = state.partial
    if event_name == "ping" or not isinstance(data, dict):
        return

    etype = data.get("type", event_name)

    if etype == "message_start":
        message = data.get("message", {})
        state.set_response(message.get("id"), message.get("model"))
        usage = message.get("usage", {}) or {}
        # Prefer the cache_creation split (5m/1h); fall back to the flat total.
        creation = usage.get("cache_creation") or {}
        write_5m = creation.get("ephemeral_5m_input_tokens")
        if write_5m is None:
            write_5m = usage.get("cache_creation_input_tokens", 0) or 0
            write_1h = 0
        else:
            write_1h = creation.get("ephemeral_1h_input_tokens", 0) or 0
        state.set_usage(
            usage_from_counts(
                input_tokens=usage.get("input_tokens", 0) or 0,
                cache_read=usage.get("cache_read_input_tokens", 0) or 0,
                cache_write=write_5m,
                cache_write_1h=write_1h,
            )
        )
        return

    if etype == "content_block_start":
        index = data["index"]
        block = data.get("content_block", {})
        block_type = block.get("type")
        if block_type == "text":
            state.add_text(index)
            yield TextStartEvent(content_index=index, partial=state.snapshot())
        elif block_type == "thinking":
            state.add_thinking(index)
            if block.get("redacted"):
                partial.content[index].redacted = True  # type: ignore[union-attr]
            yield ThinkingStartEvent(content_index=index, partial=state.snapshot())
        elif block_type == "tool_use":
            tool_call = ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments={})
            state.add_tool_call(index, tool_call)
            yield ToolCallStartEvent(content_index=index, partial=state.snapshot())
        return

    if etype == "content_block_delta":
        index = data["index"]
        delta = data.get("delta", {})
        delta_type = delta.get("type")
        block = partial.content[index] if index < len(partial.content) else None
        if delta_type == "text_delta" and isinstance(block, TextContent):
            block.text += delta.get("text", "")
            yield TextDeltaEvent(content_index=index, delta=delta.get("text", ""), partial=state.snapshot())
        elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
            block.thinking += delta.get("thinking", "")
            yield ThinkingDeltaEvent(
                content_index=index, delta=delta.get("thinking", ""), partial=state.snapshot()
            )
        elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
            block.thinking_signature = (block.thinking_signature or "") + delta.get("signature", "")
        elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
            block.arguments = {
                "__raw__": (block.arguments.get("__raw__", "") or "") + delta.get("partial_json", "")
            }
            yield ToolCallDeltaEvent(
                content_index=index, delta=delta.get("partial_json", ""), partial=state.snapshot()
            )
        return

    if etype == "content_block_stop":
        index = data["index"]
        block = partial.content[index] if index < len(partial.content) else None
        if isinstance(block, TextContent):
            yield TextEndEvent(content_index=index, content=block.text, partial=state.snapshot())
        elif isinstance(block, ThinkingContent):
            yield ThinkingEndEvent(
                content_index=index, content=block.thinking, partial=state.snapshot()
            )
        elif isinstance(block, ToolCall):
            raw = block.arguments.pop("__raw__", "") if "__raw__" in block.arguments else None
            if raw:
                try:
                    parsed = json.loads(raw)
                    block.arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
                except (json.JSONDecodeError, ValueError):
                    block.arguments = {"__invalid__": raw}
            else:
                block.arguments = {}
            yield ToolCallEndEvent(content_index=index, tool_call=block, partial=state.snapshot())
        return

    if etype == "message_delta":
        delta = data.get("delta", {})
        raw_reason = delta.get("stop_reason")
        if raw_reason:
            state.set_raw_stop_reason(raw_reason)
            state.set_end_turn(True)
        usage = data.get("usage", {}) or {}
        if usage:
            current = partial.usage
            current.output = usage.get("output_tokens", current.output) or current.output
        return

    if etype == "message_stop":
        raw = partial.raw_stop_reason or "end_turn"
        yield DoneEvent(reason=_STOP_REASON_MAP.get(raw, "stop"), message=state.snapshot())
        return

    if etype == "error":
        error = data.get("error", {})
        partial.stop_reason = "error"
        partial.error_message = f"{error.get('type', 'api_error')}: {error.get('message', 'unknown')}"
        yield StreamErrorEvent(reason="error", error=state.snapshot())
        return


async def stream_anthropic(
    model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions] = None
) -> AsyncIterator[Any]:
    """Stream an assistant response from the Anthropic Messages API."""

    import time

    timestamp = int(time.time() * 1000)
    key = (options.api_key if options and options.api_key else model.api_key) or ""
    oauth = model.auth_type == "oauth" and model.provider == "anthropic" or "sk-ant-oat" in key
    canonical = {name.lower(): name for name in ("Read", "Write", "Edit", "Bash", "Grep", "Glob", "Skill", "Task", "WebFetch", "WebSearch")}
    system, _ = normalize_context(context)
    name_map = {tool.name: canonical.get(tool.name.lower(), tool.name) for tool in (system.tools_added or [])} if system else {}
    inverse = {value: name for name, value in name_map.items()}

    def request_factory() -> Dict[str, Any]:
        body = build_request(model, context, options)
        if oauth:
            # The OAuth identity prepends a fixed first block; keep any cached
            # prompt (block list from build_request) as the trailing block.
            cached = body.get("system")
            blocks = (
                cached if isinstance(cached, list)
                else [{"type": "text", "text": cached}] if isinstance(cached, str) and cached
                else []
            )
            body["system"] = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}] + blocks
            for tool in body.get("tools", []):
                tool["name"] = name_map.get(tool["name"], tool["name"])
            for message in body["messages"]:
                for block in message["content"] if isinstance(message["content"], list) else []:
                    if block.get("type") == "tool_use":
                        block["name"] = name_map.get(block["name"], block["name"])
        return body

    retention = options.cache_retention if options else None

    def headers_factory() -> Dict[str, str]:
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
            **model.headers,
            **copilot_headers(model, context),
        }
        if oauth or model.provider == "github-copilot":
            headers["Authorization"] = "Bearer " + key
        else:
            headers["x-api-key"] = key
        if oauth:
            headers.update({"anthropic-beta": "claude-code-20250219,oauth-2025-04-20", "x-app": "cli",
                "user-agent": "claude-cli/2.1.280", "anthropic-dangerous-direct-browser-access": "true"})
        elif retention == "1h":
            # Extended cache TTL requires an opt-in beta header.
            headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
        return headers

    async def convert(event_name, data, state):
        async for event in _convert_events(event_name, data, state):
            snapshot = event.message if isinstance(event, DoneEvent) else event.error if isinstance(event, StreamErrorEvent) else event.partial
            if oauth:
                for block in snapshot.content:
                    if isinstance(block, ToolCall):
                        block.name = inverse.get(block.name, block.name)
            if isinstance(event, DoneEvent):
                snapshot.stop_reason = event.reason
                snapshot.usage.cost = model.calculate_cost(snapshot.usage)
            yield event

    async for event in run_provider_stream(
        model,
        request_factory,
        "/v1/messages",
        headers_factory,
        convert,
        options,
        timestamp,
    ):
        yield event
