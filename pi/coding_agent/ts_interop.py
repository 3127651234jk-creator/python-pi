"""TypeScript pi session interoperability.

Reads and writes the session JSONL format emitted by the TypeScript
implementation (``packages/coding-agent/src/core/session-manager.ts``,
``CURRENT_SESSION_VERSION = 3``): a leading ``{"type": "session", ...}``
header followed by ``{type, id, parentId, timestamp, ...}`` entries with
camelCase message fields.

Conversion covers the entry types that carry meaning in this port:
``message``, ``model_change``, ``thinking_level_change``, ``compaction``,
``branch_summary``, ``session_info``, ``custom`` and ``custom_message``.
Unknown entry types round-trip as opaque ``custom`` records so exporting a
file does not silently drop data.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from ..ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    Tool,
    ToolReference,
    ToolResultMessage,
    Usage,
    UserMessage,
)

TS_VERSION = 3


def _ts_usage(data: Optional[dict]) -> Usage:
    if not isinstance(data, dict):
        return Usage()
    cost = data.get("cost") or {}
    return Usage(
        input=int(data.get("input", 0) or 0),
        output=int(data.get("output", 0) or 0),
        cache_read=int(data.get("cacheRead", 0) or 0),
        cache_write=int(data.get("cacheWrite", 0) or 0),
        reasoning=data.get("reasoning") if isinstance(data.get("reasoning"), int) else None,
        cost={"input": float(cost.get("input", 0) or 0), "output": float(cost.get("output", 0) or 0),
              "cache_read": float(cost.get("cacheRead", 0) or 0), "cache_write": float(cost.get("cacheWrite", 0) or 0),
              "total": float(cost.get("total", 0) or 0)},
    )


def _our_usage(usage: Usage) -> dict:
    return {
        "input": usage.input, "output": usage.output, "cacheRead": usage.cache_read, "cacheWrite": usage.cache_write,
        "totalTokens": usage.total_tokens,
        "cost": {"input": usage.cost.get("input", 0.0), "output": usage.cost.get("output", 0.0),
                 "cacheRead": usage.cost.get("cache_read", 0.0), "cacheWrite": usage.cost.get("cache_write", 0.0),
                 "total": usage.cost.get("total", 0.0)},
        **({"reasoning": usage.reasoning} if usage.reasoning is not None else {}),
    }


def ts_message_to_ours(data: dict):
    """Convert one TS AgentMessage JSON object to this port's dataclasses."""
    role = data.get("role")
    timestamp = data.get("timestamp", 0)
    if role == "system":
        return SystemMessage(content=data.get("content", ""), timestamp=timestamp,
            sections=data.get("sections"),
            tools_added=[Tool(**t) for t in (data.get("toolsAdded") or [])] or None,
            tools_removed=[ToolReference(**t) for t in (data.get("toolsRemoved") or [])] or None)
    if role == "user":
        content = data.get("content", "")
        if isinstance(content, list):
            content = [ImageContent(data=b["data"], mime_type=b.get("mimeType", "image/png"))
                       if b.get("type") == "image" else TextContent(text=b.get("text", "")) for b in content]
        return UserMessage(content=content, timestamp=timestamp)
    if role == "assistant":
        blocks = []
        for block in data.get("content", []):
            kind = block.get("type")
            if kind == "text":
                blocks.append(TextContent(text=block.get("text", "")))
            elif kind == "thinking":
                blocks.append(ThinkingContent(thinking=block.get("thinking", ""),
                    thinking_signature=block.get("thinkingSignature"), redacted=bool(block.get("redacted"))))
            elif kind == "toolCall":
                blocks.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("arguments", {}),
                    thought_signature=block.get("thoughtSignature")))
        return AssistantMessage(content=blocks, api=data.get("api", ""), provider=data.get("provider", ""),
            model=data.get("model", ""), usage=_ts_usage(data.get("usage")),
            stop_reason=data.get("stopReason", "stop"), timestamp=timestamp,
            response_model=data.get("responseModel"), response_id=data.get("responseId"),
            thinking_level=data.get("thinkingLevel"), error_message=data.get("errorMessage"),
            raw_stop_reason=data.get("rawStopReason"), end_turn=data.get("endTurn"))
    if role == "toolResult":
        blocks = [ImageContent(data=b["data"], mime_type=b.get("mimeType", "image/png"))
                  if b.get("type") == "image" else TextContent(text=b.get("text", ""))
                  for b in data.get("content", [])]
        usage = data.get("usage")
        return ToolResultMessage(tool_call_id=data.get("toolCallId", ""), tool_name=data.get("toolName", ""),
            content=blocks, is_error=bool(data.get("isError", False)), timestamp=timestamp,
            details=data.get("details"), usage=_ts_usage(usage) if usage else None)
    raise ValueError(f"Unknown TS message role: {role!r}")


def our_message_to_ts(message) -> dict:
    """Convert this port's message dataclass to the TS AgentMessage JSON shape."""
    from ..ai.types import ToolResultMessage, get_message_text

    if isinstance(message, SystemMessage):
        result = {"role": "system", "content": message.content, "timestamp": message.timestamp}
        if message.sections is not None:
            result["sections"] = message.sections
        if message.tools_added:
            result["toolsAdded"] = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in message.tools_added]
        if message.tools_removed:
            result["toolsRemoved"] = [{"name": t.name} for t in message.tools_removed]
        return result
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return {"role": "user", "content": message.content, "timestamp": message.timestamp}
        return {"role": "user", "timestamp": message.timestamp, "content": [
            {"type": "image", "data": b.data, "mimeType": b.mime_type} if isinstance(b, ImageContent)
            else {"type": "text", "text": b.text} for b in message.content]}
    if isinstance(message, ToolResultMessage):
        result = {"role": "toolResult", "toolCallId": message.tool_call_id, "toolName": message.tool_name,
                  "content": [{"type": "image", "data": b.data, "mimeType": b.mime_type} if isinstance(b, ImageContent)
                              else {"type": "text", "text": b.text} for b in message.content],
                  "isError": message.is_error, "timestamp": message.timestamp}
        if message.details is not None:
            result["details"] = message.details
        if message.usage is not None:
            result["usage"] = _our_usage(message.usage)
        return result
    # AssistantMessage
    blocks = []
    for block in message.content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            entry = {"type": "thinking", "thinking": block.thinking}
            if block.thinking_signature:
                entry["thinkingSignature"] = block.thinking_signature
            if block.redacted:
                entry["redacted"] = True
            blocks.append(entry)
        elif isinstance(block, ToolCall):
            entry = {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}
            if block.thought_signature:
                entry["thoughtSignature"] = block.thought_signature
            blocks.append(entry)
    result = {"role": "assistant", "content": blocks, "api": message.api, "provider": message.provider,
              "model": message.model, "usage": _our_usage(message.usage), "stopReason": message.stop_reason,
              "timestamp": message.timestamp}
    for ours, ts in (("response_model", "responseModel"), ("response_id", "responseId"),
                     ("thinking_level", "thinkingLevel"), ("error_message", "errorMessage"),
                     ("raw_stop_reason", "rawStopReason"), ("end_turn", "endTurn")):
        value = getattr(message, ours)
        if value is not None:
            result[ts] = value
    return result


_CUSTOM_KIND_TO_TS = {
    "compaction": "compaction",
    "model_change": "model_change",
    "thinking_level_change": "thinking_level_change",
    "branch_summary": "branch_summary",
}
_TS_TO_CUSTOM_KIND = {value: key for key, value in _CUSTOM_KIND_TO_TS.items()}


def ts_entry_to_custom(record: dict) -> Optional[dict]:
    """Map a TS metadata entry to this port's custom-entry payload, if any."""
    kind = record.get("type")
    if kind == "compaction":
        return {"kind": "compaction", "data": {"summary": record.get("summary", ""), "tokens_before": record.get("tokensBefore", 0)}}
    if kind == "model_change":
        return {"kind": "model_change", "data": {"provider": record.get("provider", ""), "modelId": record.get("modelId", "")}}
    if kind == "thinking_level_change":
        return {"kind": "thinking_level_change", "data": {"thinkingLevel": record.get("thinkingLevel", "off")}}
    if kind == "branch_summary":
        return {"kind": "branch_summary", "data": {"fromId": record.get("fromId", ""), "summary": record.get("summary", "")}}
    return None


def custom_to_ts_entry(kind: str, data: dict) -> Optional[dict]:
    if kind == "compaction":
        return {"type": "compaction", "summary": data.get("summary", ""), "tokensBefore": data.get("tokens_before", 0)}
    if kind == "model_change":
        return {"type": "model_change", "provider": data.get("provider", ""), "modelId": data.get("modelId", "")}
    if kind == "thinking_level_change":
        return {"type": "thinking_level_change", "thinkingLevel": data.get("thinkingLevel", "off")}
    if kind == "branch_summary":
        return {"type": "branch_summary", "fromId": data.get("fromId", ""), "summary": data.get("summary", "")}
    return None


def read_ts_file(path: str) -> dict:
    """Parse a TS session file into {'name', 'records': [record...]} with messages converted.

    Each returned record is ``{'id', 'parent_id', 'message'?|'custom'?|'name'?}``.
    """
    import json

    records: List[Dict[str, Any]] = []
    name = ""
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "session":
                continue  # header
            if kind == "session_info":
                name = record.get("name", name)
                continue
            entry = {"id": record.get("id") or uuid.uuid4().hex, "parent_id": record.get("parentId")}
            if kind == "message":
                entry["message"] = ts_message_to_ours(record.get("message", {}))
            elif kind == "custom_message":
                entry["message"] = ts_message_to_ours({"role": "user", "content": record.get("content", ""), "timestamp": record.get("timestamp", 0)})
                entry["custom"] = {"kind": record.get("customType", "custom_message"), "data": {"display": record.get("display", True)}}
            elif kind == "custom":
                entry["custom"] = {"kind": record.get("customType", ""), "data": record.get("data", {})}
            elif kind in _TS_TO_CUSTOM_KIND:
                entry["custom"] = ts_entry_to_custom(record)
            else:
                # Unknown TS entry types (usage, context_edit, label, ...) round-trip opaquely.
                entry["custom"] = {"kind": "ts_" + str(kind), "data": record}
            records.append(entry)
    return {"name": name, "records": records}


def write_ts_file(path: str, session) -> str:
    """Write a SessionManager's active branch in the TS session format."""
    import json

    header = {"type": "session", "version": TS_VERSION, "id": str(uuid.uuid4()),
              "timestamp": int(time.time() * 1000), "cwd": session.cwd}
    lines = [json.dumps(header, ensure_ascii=False)]
    if session.name:
        lines.append(json.dumps({"type": "session_info", "id": uuid.uuid4().hex, "parentId": None,
                                 "timestamp": int(time.time() * 1000), "name": session.name}, ensure_ascii=False))
    for entry in session.get_branch():
        common = {"id": entry.id, "parentId": entry.parent_id}
        if entry.message is not None and entry.custom is None:
            lines.append(json.dumps({**common, "type": "message", "timestamp": entry.message.timestamp,
                                     "message": our_message_to_ts(entry.message)}, ensure_ascii=False))
        elif entry.message is not None and entry.custom is not None:
            lines.append(json.dumps({**common, "type": "custom_message", "customType": entry.custom["kind"],
                                     "display": entry.custom["data"].get("display", True),
                                     "timestamp": entry.message.timestamp,
                                     "content": entry.message.content}, ensure_ascii=False))
        elif entry.custom is not None:
            mapped = custom_to_ts_entry(entry.custom["kind"], entry.custom["data"])
            if mapped is not None:
                lines.append(json.dumps({**common, **mapped, "timestamp": int(time.time() * 1000)}, ensure_ascii=False))
            elif entry.custom["kind"].startswith("ts_"):
                record = dict(entry.custom["data"])
                record.update(common)
                lines.append(json.dumps(record, ensure_ascii=False))
            else:
                lines.append(json.dumps({**common, "type": "custom", "customType": entry.custom["kind"],
                                         "data": entry.custom["data"], "timestamp": int(time.time() * 1000)}, ensure_ascii=False))
    with open(path, "x", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))
    return path
