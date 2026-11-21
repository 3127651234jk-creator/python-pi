"""Faux provider: scripted responses for tests and offline runs.

Python port of ``packages/ai/src/providers/faux.ts``. The script is a list of
turns; each request consumes the next one:

    {"text": "hello"}                                  -> plain response
    {"thinking": "...", "text": "..."}                 -> thinking + text
    {"toolCalls": [{"id","name","arguments"}], "text": "..."} -> toolUse turn
    {"error": "message"}                               -> error turn

The script can be passed via ``options.script`` or the ``FAUX_SCRIPT``
environment variable (JSON-encoded list).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import Model
from ..types import (
    AssistantMessage,
    DoneEvent,
    ImageContent,
    StreamErrorEvent,
    StreamStartEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    TranscriptContext,
    Usage,
    get_system_message_text,
)
from ..utils import normalize_context
from .base import SimpleStreamOptions


def default_script() -> List[Dict[str, Any]]:
    return [{"text": "Faux response."}]


async def stream_faux(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AsyncIterator[Any]:
    options = options or SimpleStreamOptions()
    script = getattr(options, "script", None) or json.loads(os.environ.get("FAUX_SCRIPT", "[]")) or default_script()
    script_index = _next_index()
    turn = script[script_index] if script_index < len(script) else {"error": "Faux script exhausted"}
    timestamp = int(time.time() * 1000)

    partial = AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=timestamp,
    )
    partial.thinking_level = options.reasoning

    if turn.get("error"):
        partial.stop_reason = "error"
        partial.error_message = str(turn["error"])
        yield StreamStartEvent(partial=partial)
        yield StreamErrorEvent(reason="error", error=partial)
        return

    yield StreamStartEvent(partial=partial)
    index = 0

    thinking_text = turn.get("thinking")
    if thinking_text:
        block = ThinkingContent(thinking="")
        partial.content.append(block)
        yield ThinkingStartEvent(content_index=index, partial=partial)
        block.thinking = thinking_text
        yield ThinkingDeltaEvent(content_index=index, delta=thinking_text, partial=partial)
        yield ThinkingEndEvent(content_index=index, content=thinking_text, partial=partial)
        index += 1

    text = turn.get("text", "")
    if text:
        block = TextContent(text="")
        partial.content.append(block)
        yield TextStartEvent(content_index=index, partial=partial)
        block.text = text
        yield TextDeltaEvent(content_index=index, delta=text, partial=partial)
        yield TextEndEvent(content_index=index, content=text, partial=partial)
        index += 1

    tool_calls = turn.get("toolCalls") or []
    for raw in tool_calls:
        tool_call = ToolCall(id=raw.get("id", f"call_{index}"), name=raw.get("name", ""), arguments={})
        partial.content.append(tool_call)
        yield ToolCallStartEvent(content_index=index, partial=partial)
        args_json = json.dumps(raw.get("arguments", {}))
        tool_call.arguments = {"__raw__": args_json}
        yield ToolCallDeltaEvent(content_index=index, delta=args_json, partial=partial)
        tool_call.arguments = raw.get("arguments", {})
        yield ToolCallEndEvent(content_index=index, tool_call=tool_call, partial=partial)
        index += 1

    # A scripted "stopReason" overrides the derived one, mirroring the TS
    # fauxAssistantMessage options (e.g. stopReason "length" for truncation).
    stop_reason = turn.get("stopReason") or ("toolUse" if tool_calls else "stop")
    partial.stop_reason = stop_reason  # type: ignore[assignment]
    partial.usage = _usage_estimate(partial, context, options)
    yield DoneEvent(reason=stop_reason, message=partial)  # type: ignore[arg-type]


_INDEX = 0
# Per-session previous prompts for the cache-usage estimate; reset with the
# script index so tests never observe another test's session.
_PROMPT_CACHE: Dict[str, List[str]] = {}


def _next_index() -> int:
    global _INDEX
    index = _INDEX
    _INDEX += 1
    return index


def reset_script_index() -> None:
    global _INDEX
    _INDEX = 0
    _PROMPT_CACHE.clear()


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(f"[image:{block.mime_type}:{len(block.data)}]"
                     if isinstance(block, ImageContent) else block.text for block in content)


def _assistant_content_to_text(content) -> str:
    return "\n".join(
        block.thinking if isinstance(block, ThinkingContent)
        else f"{block.name}:{json.dumps(block.arguments)}" if isinstance(block, ToolCall)
        else block.text
        for block in content)


def _message_to_text(message) -> str:
    if message.role == "system":
        parts = [get_system_message_text(message)]
        parts += [f"tool-:{json.dumps(asdict(tool))}" for tool in (message.tools_removed or [])]
        parts += [f"tool+:{json.dumps(asdict(tool))}" for tool in (message.tools_added or [])]
        return "\n".join(part for part in parts if part)
    if message.role == "user":
        return _content_to_text(message.content)
    if message.role == "assistant":
        return _assistant_content_to_text(message.content)
    return "\n".join([message.tool_name] + [_content_to_text([block]) for block in message.content])


def _joined_length(messages: List[str], count: Optional[int] = None) -> int:
    count = len(messages) if count is None else count
    length = (count - 1) * 2 if count > 0 else 0
    for index in range(count):
        length += len(messages[index])
    return length


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def _common_prompt_prefix_length(previous: List[str], current: List[str]) -> int:
    """Length of the common prefix of the two joined prompts. Equal messages
    are compared whole; characters are compared only from the first message
    that differs (keeps the estimate independent of message boundaries)."""
    index = 0
    while index < len(previous) and index < len(current) and previous[index] == current[index]:
        index += 1

    def rest(messages: List[str]) -> str:
        if index == len(messages):
            return ""
        return ("\n\n" if index > 0 else "") + "\n\n".join(messages[index:])

    return _joined_length(previous, index) + _common_prefix_length(rest(previous), rest(current))


def _usage_estimate(message: AssistantMessage, context: TranscriptContext,
                    options: Optional[SimpleStreamOptions]) -> Usage:
    """Estimate the response usage like the TS faux provider: prompt tokens
    from the serialized context, output tokens from the response content, and
    a per-session prompt-cache split into cacheRead/cacheWrite."""
    prompt = [f"{item.role}:{_message_to_text(item)}" for item in context.messages]
    prompt_length = _joined_length(prompt)
    prompt_tokens = math.ceil(prompt_length / 4)
    output_tokens = math.ceil(len(_assistant_content_to_text(message.content)) / 4)
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = getattr(options, "session_id", None) if options is not None else None
    if session_id and getattr(options, "cache_retention", None) != "none":
        previous = _PROMPT_CACHE.get(session_id)
        if previous is not None:
            cached_chars = _common_prompt_prefix_length(previous, prompt)
            cache_read = math.ceil(cached_chars / 4)
            cache_write = math.ceil((prompt_length - cached_chars) / 4)
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        _PROMPT_CACHE[session_id] = prompt
    return Usage(input=input_tokens, output=output_tokens, cache_read=cache_read, cache_write=cache_write)


def declared_tools(context: TranscriptContext) -> List[Any]:
    system, _ = normalize_context(context)
    return list(system.tools_added or []) if system else []
