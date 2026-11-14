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
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import Model
from ..types import (
    AssistantMessage,
    DoneEvent,
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

    stop_reason = "toolUse" if tool_calls else "stop"
    partial.stop_reason = stop_reason  # type: ignore[assignment]
    partial.usage = Usage(input=100, output=50)
    yield DoneEvent(reason=stop_reason, message=partial)  # type: ignore[arg-type]


_INDEX = 0


def _next_index() -> int:
    global _INDEX
    index = _INDEX
    _INDEX += 1
    return index


def reset_script_index() -> None:
    global _INDEX
    _INDEX = 0


def declared_tools(context: TranscriptContext) -> List[Any]:
    system, _ = normalize_context(context)
    return list(system.tools_added or []) if system else []
