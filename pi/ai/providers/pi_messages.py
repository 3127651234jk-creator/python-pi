"""Radius /messages wire protocol, with camelCase transcript and SSE events."""

import json
import time
from dataclasses import fields

from .. import types
from ..json_io import to_json, message_from_json
from .base import SimpleStreamOptions, run_provider_stream


WIRE_NAMES = {
    "mime_type": "mimeType", "tools_added": "toolsAdded", "tools_removed": "toolsRemoved",
    "tool_call_id": "toolCallId", "tool_name": "toolName", "is_error": "isError",
    "thinking_signature": "thinkingSignature", "thought_signature": "thoughtSignature",
    "stop_reason": "stopReason", "response_id": "responseId", "response_model": "responseModel",
    "thinking_level": "thinkingLevel", "error_message": "errorMessage", "raw_stop_reason": "rawStopReason",
    "end_turn": "endTurn", "cache_read": "cacheRead", "cache_write": "cacheWrite",
    "cache_write_1h": "cacheWrite1h",
}
EVENTS = {cls.type: cls for cls in (
    types.TextStartEvent, types.TextDeltaEvent, types.TextEndEvent,
    types.ThinkingStartEvent, types.ThinkingDeltaEvent, types.ThinkingEndEvent,
    types.ToolCallStartEvent, types.ToolCallDeltaEvent, types.ToolCallEndEvent)}


def wire_message(message):
    data = to_json(message)
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {WIRE_NAMES.get(key, key): (item if key in ("arguments", "parameters", "details", "sections")
                                         else convert(item)) for key, item in value.items() if item is not None}
    return convert(data)


def build_request(model, context, options):
    values = {"temperature": options.temperature, "maxTokens": options.max_tokens,
              "reasoning": options.reasoning,
              "cacheRetention": {"5m": "short", "1h": "long"}.get(options.cache_retention)}
    return {"model": model.id, "context": {"messages": [wire_message(m) for m in context.messages]},
            "options": {key: value for key, value in values.items() if value is not None}}


async def stream_pi_messages(model, context, options=None):
    options = options or SimpleStreamOptions()
    raw = {"role": "assistant", "content": [], "api": model.api, "provider": model.provider,
           "model": model.id, "timestamp": int(time.time() * 1000)}
    arguments = {}
    reverse = {value: key for key, value in WIRE_NAMES.items()}

    def snapshot():
        data = dict(raw)
        data["content"] = [{reverse.get(k, k): v for k, v in block.items()} for block in raw["content"]]
        usage = raw.get("usage", {})
        data["usage"] = {reverse.get(k, k): v for k, v in usage.items() if k != "totalTokens"}
        if "cost" in data["usage"]:
            data["usage"]["cost"] = {reverse.get(k, k): v for k, v in usage["cost"].items()}
        return message_from_json(data)

    async def convert(event_name, event, state):
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "start":
            return  # shared driver already emitted start
        if kind in ("done", "error"):
            raw.update(usage=event.get("usage", {}), stop_reason=event["reason"],
                       response_id=event.get("responseId"), error_message=event.get("errorMessage"))
            message = snapshot()
            if kind == "done":
                yield types.DoneEvent(reason=event["reason"], message=message)
            else:
                yield types.StreamErrorEvent(reason=event["reason"], error=message)
            return
        if kind not in EVENTS:
            raise ValueError(f"Unknown pi-messages event: {kind}")
        index = event["contentIndex"]
        content = raw["content"]
        if kind.endswith("_start"):
            if index != len(content):
                raise ValueError("Nonsequential content index")
            if kind == "toolcall_start":
                content.append({"type": "toolCall", "id": event["id"], "name": event["toolName"], "arguments": {}})
                arguments[index] = ""
            else:
                field = "text" if kind == "text_start" else "thinking"
                content.append({"type": field, field: ""})
        elif kind == "toolcall_delta":
            arguments[index] += event["delta"]
            try:
                content[index]["arguments"] = json.loads(arguments[index])
            except ValueError:
                pass
        elif kind == "toolcall_end":
            content[index] = event["toolCall"]
        else:
            field = "text" if kind.startswith("text_") else "thinking"
            if kind.endswith("_delta"):
                content[index][field] += event["delta"]
            else:
                content[index][field] = event["content"]
                if field == "thinking":
                    content[index]["thinkingSignature"] = event.get("contentSignature")
                    content[index]["redacted"] = event.get("redacted", False)
        state._partial = snapshot()
        values = {"content_index": index, "partial": state.snapshot(), "delta": event.get("delta"),
                  "content": event.get("content"), "tool_call": state.partial.content[index]}
        yield EVENTS[kind](**{f.name: values[f.name] for f in fields(EVENTS[kind]) if f.init})

    async for event in run_provider_stream(model, lambda: build_request(model, context, options), "/messages",
        lambda: {"Authorization": "Bearer " + (options.api_key or model.api_key or ""),
                 "Accept": "text/event-stream", "Content-Type": "application/json", **model.headers},
        convert, options, int(time.time() * 1000)):
        yield event
