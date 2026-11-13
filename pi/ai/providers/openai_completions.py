"""openai-completions provider adapter (streaming SSE).

Python port of ``packages/ai/src/api/openai-completions.ts``: works with the
OpenAI Chat Completions API and OpenAI-compatible endpoints (DeepSeek, Groq,
OpenRouter, vLLM, llama.cpp, ...).
"""

from __future__ import annotations

import json
from urllib.parse import quote, urlencode, urlsplit
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import Model
from ..types import (
    AssistantMessage,
    AudioContent,
    DoneEvent,
    ImageContent,
    StreamErrorEvent,
    SystemMessage,
    ThinkingContent,
    Tool,
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
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
    VideoContent,
    get_system_message_text,
)
from ..utils import normalize_context
from .copilot import copilot_headers
from .base import (
    ProviderState,
    SimpleStreamOptions,
    apply_model_params,
    run_provider_stream,
    usage_from_counts,
)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "tool_calls": "toolUse",
    "function_call": "toolUse",
    "length": "length",
    "max_tokens": "length",
}


def build_request(model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions]) -> Dict[str, Any]:
    """Build the ``/chat/completions`` payload from a normalized transcript."""
    system, messages = normalize_context(context)
    body: Dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": convert_messages(system, messages),
    }
    if model.compat.get("supportsUsageInStreaming", True):
        body["stream_options"] = {"include_usage": True}
    default_token_field = (
        "max_completion_tokens" if model.provider == "openai" or model.api == "azure-openai"
        or urlsplit(model.base_url).hostname == "api.openai.com" else "max_tokens"
    )
    token_field = model.compat.get("maxTokensField", default_token_field)
    if token_field not in ("max_tokens", "max_completion_tokens"):
        raise ValueError("compat.maxTokensField must be max_tokens or max_completion_tokens")
    tools = _declared_tools(system)
    if tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
            if not t.hosted
        ]
        body["tool_choice"] = "auto"
        if not body["tools"]:
            del body["tools"]
            del body["tool_choice"]
    if options and options.max_tokens:
        body[token_field] = options.max_tokens
    else:
        body[token_field] = model.max_tokens
    if options and options.temperature is not None:
        body["temperature"] = options.temperature
    reasoning = options.reasoning if options else None
    if model.reasoning and reasoning not in (None, "off"):
        body["reasoning_effort"] = _reasoning_effort(reasoning)
    apply_model_params(body, model, options)
    if model.api == "azure-openai":
        body.pop("api_version", None)
    return body


def _reasoning_effort(level: Optional[str]) -> str:
    return {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}.get(
        level or "medium", "medium"
    )


def _declared_tools(system: Optional[SystemMessage]) -> List[Tool]:
    if system is None or not system.tools_added:
        return []
    return list(system.tools_added)


def convert_messages(system: Optional[SystemMessage], messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert the transcript to OpenAI chat messages."""
    api_messages: List[Dict[str, Any]] = []
    if system is not None:
        text = get_system_message_text(system)
        if text:
            api_messages.append({"role": "system", "content": text})

    for message in messages:
        if isinstance(message, UserMessage):
            api_messages.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            api_messages.append(_assistant_message(message))
        elif isinstance(message, ToolResultMessage):
            text = "\n".join(b.text for b in message.content if isinstance(b, TextContent)) or "(empty result)"
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": text,
                }
            )
    return api_messages


# Chat Completions audio input accepts only these formats, keyed by MIME type.
_AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}


def _user_content(message: UserMessage) -> Any:
    if isinstance(message.content, str):
        return message.content
    parts: List[Dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ImageContent):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
        elif isinstance(block, AudioContent):
            fmt = _AUDIO_FORMATS.get(block.mime_type.split(";")[0].strip().lower())
            if fmt is None:
                raise ValueError(f"Unsupported audio type {block.mime_type}: OpenAI accepts audio/wav and audio/mpeg")
            parts.append({"type": "input_audio", "input_audio": {"data": block.data, "format": fmt}})
        elif isinstance(block, VideoContent):
            raise ValueError("OpenAI Chat Completions does not support video input")
        else:
            parts.append({"type": "text", "text": block.text})
    return parts


def _assistant_message(message: AssistantMessage) -> Dict[str, Any]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, ToolCall) and block.raw_item is None:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                }
            )
        # Thinking blocks are dropped: the completions API has no replay field.
    result: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _tool_call_blocks(partial: AssistantMessage) -> List[ToolCall]:
    return [b for b in partial.content if isinstance(b, ToolCall)]


async def _convert_events(event_name: Optional[str], data: Any, state: ProviderState) -> AsyncIterator[Any]:
    partial = state.partial

    if data == "[DONE]":
        if partial.raw_stop_reason is None:
            # Some gateways end with [DONE] but never send finish_reason. Infer
            # the reason and close the blocks; otherwise tool arguments stay as
            # {"__raw__": ...} and every call fails while the model retries.
            has_tool_calls = any(isinstance(block, ToolCall) for block in partial.content)
            partial.stop_reason = "toolUse" if has_tool_calls else "stop"
            for event in _close_blocks(state):
                yield event
        yield DoneEvent(reason=partial.stop_reason, message=state.snapshot())
        return
    if not isinstance(data, dict):
        return

    if "error" in data and data.get("error"):
        error = data["error"]
        partial.stop_reason = "error"
        partial.error_message = str(error.get("message", error) if isinstance(error, dict) else error)
        yield StreamErrorEvent(reason="error", error=state.snapshot())
        return

    if data.get("usage"):
        usage = data["usage"]
        details = usage.get("prompt_tokens_details", {}) or {}
        completion_details = usage.get("completion_tokens_details", {}) or {}
        # DeepSeek reports prompt_cache_hit_tokens; Kimi a top-level cached_tokens.
        cached = (
            details.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or usage.get("cached_tokens")
            or 0
        )
        counted = usage_from_counts(
            # prompt_tokens includes cache reads; report them separately.
            input_tokens=max(0, (usage.get("prompt_tokens", 0) or 0) - cached),
            output_tokens=usage.get("completion_tokens", 0) or 0,
            cache_read=cached,
            reasoning=completion_details.get("reasoning_tokens"),
        )
        # Usage arrives in its own final chunk; the [DONE] handler skips the
        # shared cost application, so price it here. Keep going: some gateways
        # attach usage to the same chunk that carries finish_reason.
        counted.cost = state.model.calculate_cost(counted)
        state.set_usage(counted)

    choices = data.get("choices") or []
    if not choices:
        return
    choice = choices[0]
    state.set_response(data.get("id"), data.get("model"))

    delta = choice.get("delta") or {}
    if delta.get("refusal"):
        raise RuntimeError(f"Provider refusal: {delta['refusal']}")

    # Reasoning content (DeepSeek-style) maps to thinking blocks.
    reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
    if reasoning_text:
        thinking = _ensure_thinking(partial)
        index = partial.content.index(thinking)
        was_start = thinking.thinking == ""
        thinking.thinking += reasoning_text
        if was_start:
            yield ThinkingStartEvent(content_index=index, partial=state.snapshot())
        yield ThinkingDeltaEvent(content_index=index, delta=reasoning_text, partial=state.snapshot())

    content_text = delta.get("content")
    if content_text:
        text = _ensure_text(partial)
        index = partial.content.index(text)
        was_start = text.text == ""
        text.text += content_text
        if was_start:
            yield TextStartEvent(content_index=index, partial=state.snapshot())
        yield TextDeltaEvent(content_index=index, delta=content_text, partial=state.snapshot())

    for entry in delta.get("tool_calls") or []:
        index = entry.get("index", 0)
        calls = _tool_call_blocks(partial)
        if index >= len(calls):
            function = entry.get("function") or {}
            tool_call = ToolCall(
                id=entry.get("id", f"call_{index}"),
                name=function.get("name", ""),
                arguments={"__raw__": ""},
            )
            state.add_tool_call(len(partial.content), tool_call)
            yield ToolCallStartEvent(content_index=len(partial.content) - 1, partial=state.snapshot())
            calls = _tool_call_blocks(partial)
        block = calls[index]
        function = entry.get("function") or {}
        if entry.get("id"):
            block.id = entry["id"]
        if function.get("name"):
            block.name = function["name"]
        args_delta = function.get("arguments")
        if args_delta:
            block.arguments["__raw__"] = block.arguments.get("__raw__", "") + args_delta
            yield ToolCallDeltaEvent(
                content_index=partial.content.index(block), delta=args_delta, partial=state.snapshot()
            )

    finish_reason = choice.get("finish_reason")
    if finish_reason:
        if partial.raw_stop_reason is not None:
            return
        state.set_raw_stop_reason(finish_reason)
        partial.stop_reason = _FINISH_REASON_MAP.get(finish_reason, "error")
        if partial.stop_reason == "error":
            raise RuntimeError(f"Provider finish_reason: {finish_reason}")
        for event in _close_blocks(state):
            yield event
        return


def _close_blocks(state: ProviderState) -> List[Any]:
    """End events for open blocks, parsing accumulated tool-call arguments."""
    partial = state.partial
    events: List[Any] = []
    for index, block in enumerate(list(partial.content)):
        if isinstance(block, TextContent):
            events.append(TextEndEvent(content_index=index, content=block.text, partial=state.snapshot()))
        elif isinstance(block, ThinkingContent):
            events.append(ThinkingEndEvent(content_index=index, content=block.thinking, partial=state.snapshot()))
        elif isinstance(block, ToolCall):
            if "__raw__" not in block.arguments:
                continue  # Already finalized: retain parsed arguments.
            raw = block.arguments.pop("__raw__")
            if raw:
                try:
                    parsed = json.loads(raw)
                    block.arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
                except (json.JSONDecodeError, ValueError):
                    block.arguments = {"__invalid__": raw}
            else:
                block.arguments = {}
            events.append(ToolCallEndEvent(content_index=index, tool_call=block, partial=state.snapshot()))
    return events


def _ensure_text(partial: AssistantMessage) -> TextContent:
    for block in reversed(partial.content):
        if isinstance(block, TextContent):
            return block
    text = TextContent(text="")
    partial.content.append(text)
    return text


def _ensure_thinking(partial: AssistantMessage) -> ThinkingContent:
    for block in reversed(partial.content):
        if isinstance(block, ThinkingContent):
            return block
    thinking = ThinkingContent(thinking="")
    partial.content.append(thinking)
    return thinking


async def stream_openai_completions(
    model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions] = None
) -> AsyncIterator[Any]:
    """Stream an assistant response from an OpenAI-compatible endpoint.

    ``azure-openai`` uses the same payload but Azure-specific transport: the
    deployment name rides in the URL path and the key goes in ``api-key``.
    """

    import os
    import time

    timestamp = int(time.time() * 1000)
    azure = model.api == "azure-openai"
    key = (options.api_key if options and options.api_key else model.api_key) or ""

    def request_factory() -> Dict[str, Any]:
        return build_request(model, context, options)

    def headers_factory() -> Dict[str, str]:
        if azure:
            if model.auth_type == "entra":
                token = key or os.environ.get("AZURE_ACCESS_TOKEN", "")
                return {"Authorization": f"Bearer {token}", "content-type": "application/json", **model.headers}
            return {"api-key": key, "content-type": "application/json", **model.headers}
        return {
            "Authorization": f"Bearer {key}",
            "content-type": "application/json",
            **model.headers,
            **copilot_headers(model, context),
        }

    if azure:
        version = model.params.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-21"
        path = f"/openai/deployments/{quote(model.id, safe='')}/chat/completions?{urlencode({'api-version': version})}"
    else:
        path = "/chat/completions"

    async for event in run_provider_stream(
        model,
        request_factory,
        path,
        headers_factory,
        _convert_events,
        options,
        timestamp,
    ):
        yield event
