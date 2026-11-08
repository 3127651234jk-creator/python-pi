"""Radius /messages wire protocol, with camelCase transcript and SSE events."""

import json
import os
import time
from dataclasses import fields
from typing import Any, Dict, Optional

from .. import types
from ..json_io import to_json, message_from_json
from .base import SimpleStreamOptions, run_provider_stream


WIRE_NAMES = {
    "mime_type": "mimeType", "tools_added": "toolsAdded", "tools_removed": "toolsRemoved",
    "tool_call_id": "toolCallId", "tool_name": "toolName", "is_error": "isError",
    "thinking_signature": "thinkingSignature", "thought_signature": "thoughtSignature",
    "text_signature": "textSignature", "provider_thinking_level": "providerThinkingLevel",
    "stop_reason": "stopReason", "response_id": "responseId", "response_model": "responseModel",
    "thinking_level": "thinkingLevel", "error_message": "errorMessage", "raw_stop_reason": "rawStopReason",
    "end_turn": "endTurn", "cache_read": "cacheRead", "cache_write": "cacheWrite",
    "cache_write_1h": "cacheWrite1h",
}
EVENTS = {cls.type: cls for cls in (
    types.TextStartEvent, types.TextDeltaEvent, types.TextEndEvent,
    types.ThinkingStartEvent, types.ThinkingDeltaEvent, types.ThinkingEndEvent,
    types.ToolCallStartEvent, types.ToolCallDeltaEvent, types.ToolCallEndEvent)}


class PiMessagesResponseError(Exception):
    """Non-2xx backend response, carrying redacted diagnostic details (v1)."""

    def __init__(self, message: str, code: Optional[str], diagnostic_details: Dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic_details = diagnostic_details


def parse_error_body(body: str) -> Optional[Dict[str, Any]]:
    """Only JSON objects with an object ``error`` field describe a backend error."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("error"), dict):
        return None
    return parsed


def truncate_diagnostic_string(value: str, max_length: int = 8192) -> str:
    return value if len(value) <= max_length else value[:max_length] + "…"


def create_response_error(model, url: str, status: int, status_text: str, body: str) -> PiMessagesResponseError:
    error_body = parse_error_body(body)
    error = error_body["error"] if error_body else None
    message = error.get("message") if isinstance(error, dict) and isinstance(error.get("message"), str) else None
    code = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
    suffix = message if message is not None else body
    code_suffix = f" ({code})" if code else ""
    details: Dict[str, Any] = {"version": 1, "provider": model.provider, "model": model.id, "url": url,
                               "status": status, "statusText": status_text}
    if error is not None:
        details["error"] = error
    else:
        details["body"] = truncate_diagnostic_string(body)
    details["timestampMs"] = int(time.time() * 1000)
    return PiMessagesResponseError(f"{status} {status_text}: {suffix}{code_suffix}", code, details)


def wire_message(message):
    data = to_json(message)
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        # nested_calls is a session record, not model context.
        return {WIRE_NAMES.get(key, key): (item if key in ("arguments", "parameters", "details", "sections")
                                         else convert(item))
                for key, item in value.items() if item is not None and key != "nested_calls"}
    return convert(data)


def resolve_cache_retention(value: Optional[str]) -> Optional[str]:
    if value:
        return {"5m": "short", "1h": "long"}.get(value)
    # Backend defaults apply when unset; only the legacy env opt-in is mapped.
    return "long" if os.environ.get("PI_CACHE_RETENTION") == "long" else None


def build_request(model, context, options):
    values = {"temperature": options.temperature, "maxTokens": options.max_tokens,
              "reasoning": options.reasoning,
              "cacheRetention": resolve_cache_retention(options.cache_retention),
              "sessionId": options.session_id, "toolChoice": options.tool_choice}
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
                       response_id=event.get("responseId"), error_message=event.get("errorMessage"),
                       provider_thinking_level=event.get("providerThinkingLevel"))
            message = snapshot()
            rewrite = event.get("rewrite")
            if rewrite:
                types.append_assistant_message_diagnostic(message, types.AssistantMessageDiagnostic(
                    type="pi_messages_rewrite", timestamp=int(time.time() * 1000), details=dict(rewrite)))
            if kind == "done":
                yield types.DoneEvent(reason=event["reason"], message=message)
            else:
                yield types.StreamErrorEvent(reason=event["reason"], error=message)
            return
        if kind not in EVENTS:
            # Unknown wire events are forwarded as-is with the partial attached,
            # like the TS converter, so new backend events are not fatal.
            yield {**event, "partial": state.snapshot()}
            return
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
                if field == "text":
                    content[index]["textSignature"] = event.get("contentSignature")
                else:
                    content[index]["thinkingSignature"] = event.get("contentSignature")
                    content[index]["redacted"] = event.get("redacted", False)
        state._partial = snapshot()
        values = {"content_index": index, "partial": state.snapshot(), "delta": event.get("delta"),
                  "content": event.get("content"), "tool_call": state.partial.content[index]}
        yield EVENTS[kind](**{f.name: values[f.name] for f in fields(EVENTS[kind]) if f.init})

    def response_error(status, status_text, url, body):
        error = create_response_error(model, url, status, status_text, body)
        diagnostic = types.create_assistant_message_diagnostic("pi_messages_response_failure", error,
                                                               error.diagnostic_details)
        return str(error), diagnostic

    endpoint = "/messages?debug=1" if options.debug else "/messages"
    async for event in run_provider_stream(model, lambda: build_request(model, context, options), endpoint,
        lambda: {"Authorization": "Bearer " + (options.api_key or model.api_key or ""),
                 "Accept": "text/event-stream", "Content-Type": "application/json", **model.headers},
        convert, options, int(time.time() * 1000), on_response_error=response_error):
        yield event
