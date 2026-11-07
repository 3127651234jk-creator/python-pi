"""TS wire-format converters for the JSONL RPC protocol.

Python internals keep snake_case dataclasses; every record crossing the RPC
boundary is converted here to the exact JSON shapes emitted by
``packages/coding-agent/src/modes/rpc`` (camelCase fields, optional keys
omitted when unset). Conversion is type-dispatched: opaque dicts (tool
arguments, custom entry data, TS session records) pass through untouched so
provider payloads are never rewritten.

Model JSON drops the Python-only fields (``api_key``, ``auth_type``,
``params``, ``get_api_key``, ``output``); TS ``ModelCost`` has no 1h tier, so
``cache_write_1h`` is not emitted either.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..ai.models import Model, ModelInputLimits
from ..ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from ..agent.types import AgentToolResult
from .session import SessionEntry
from .ts_interop import custom_to_ts_entry, our_message_to_ts, usage_to_ts


def model_to_ts(model: Model) -> dict:
    result = {
        "id": model.id,
        "name": model.name,
        "api": model.api,
        "provider": model.provider,
        "baseUrl": model.base_url,
        "input": list(model.input),
    }
    limits = input_limits_to_ts(model.input_limits)
    if limits is not None:
        result["inputLimits"] = limits
    result["cost"] = {
        "input": model.costs.input,
        "output": model.costs.output,
        "cacheRead": model.costs.cache_read,
        "cacheWrite": model.costs.cache_write,
    }
    if model.headers:
        result["headers"] = dict(model.headers)
    result["reasoning"] = model.reasoning
    result["contextWindow"] = model.context_window
    result["maxTokens"] = model.max_tokens
    return result


def input_limits_to_ts(limits: Optional[ModelInputLimits]) -> Optional[dict]:
    if limits is None:
        return None
    result: Dict[str, Any] = {}
    if limits.max_request_bytes is not None:
        result["maxRequestBytes"] = limits.max_request_bytes
    if limits.images is not None:
        images: Dict[str, Any] = {}
        resize = limits.images.resize
        if resize is not None:
            shaped = {key: value for key, value in (
                ("maxWidth", resize.max_width),
                ("maxHeight", resize.max_height),
                ("maxBytes", resize.max_bytes),
                ("jpegQuality", resize.jpeg_quality),
            ) if value is not None}
            if shaped:
                images["resize"] = shaped
        if limits.images.max_per_message is not None:
            images["maxPerMessage"] = limits.images.max_per_message
        if limits.images.max_per_request is not None:
            images["maxPerRequest"] = limits.images.max_per_request
        result["images"] = images
    return result


def tool_call_to_ts(block: ToolCall) -> dict:
    result = {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}
    if block.thought_signature:
        result["thoughtSignature"] = block.thought_signature
    return result


def content_blocks_to_ts(content) -> list:
    return [
        {"type": "image", "data": block.data, "mimeType": block.mime_type}
        if isinstance(block, ImageContent) else {"type": "text", "text": block.text}
        for block in content
    ]


def message_to_ts(message) -> dict:
    """One transcript message in the TS AgentMessage JSON shape."""
    return our_message_to_ts(message)


def tool_result_to_ts(value: Any) -> Any:
    """Camelize an AgentToolResult; foreign payloads pass through verbatim."""
    if isinstance(value, AgentToolResult):
        result: Dict[str, Any] = {"content": content_blocks_to_ts(value.content)}
        if value.details is not None:
            result["details"] = value.details
        if value.structured_content is not None:
            result["structuredContent"] = value.structured_content
        if value.usage is not None:
            result["usage"] = usage_to_ts(value.usage)
        result["isError"] = value.is_error
        result["terminate"] = value.terminate
        return result
    return value


def stream_event_to_ts(event, message) -> dict:
    """assistantMessageEvent with the cumulative partial snapshot stripped."""
    kind = event.type
    if kind == "start":
        return {"type": "start"}
    if kind == "toolcall_start":
        block = message.content[event.content_index]
        if not isinstance(block, ToolCall):
            raise ValueError(f"toolcall_start content at index {event.content_index} is not a tool call")
        return {"type": "toolcall_start", "contentIndex": event.content_index,
                "id": block.id, "toolName": block.name}
    if kind in ("text_start", "thinking_start"):
        return {"type": kind, "contentIndex": event.content_index}
    if kind in ("text_delta", "thinking_delta", "toolcall_delta"):
        return {"type": kind, "contentIndex": event.content_index, "delta": event.delta}
    if kind in ("text_end", "thinking_end"):
        return {"type": kind, "contentIndex": event.content_index, "content": event.content}
    if kind == "toolcall_end":
        return {"type": "toolcall_end", "contentIndex": event.content_index,
                "toolCall": tool_call_to_ts(event.tool_call)}
    if kind == "done":
        return {"type": "done", "reason": event.reason, "message": message_to_ts(event.message)}
    if kind == "error":
        return {"type": "error", "reason": event.reason, "error": message_to_ts(event.error)}
    raise ValueError(f"Unknown assistant message event: {kind!r}")


def message_update_to_ts(event) -> dict:
    message = event.message
    return {
        "type": "message_update",
        "usage": usage_to_ts(message.usage),
        "assistantMessageEvent": stream_event_to_ts(event.assistant_message_event, message),
    }


def event_to_ts(event) -> dict:
    """One AgentSessionEvent in the TS JSON shape (json-event.ts rules)."""
    kind = event.type
    if kind == "message_update":
        return message_update_to_ts(event)
    if kind in ("agent_start", "turn_start"):
        return {"type": kind}
    if kind == "turn_end":
        return {"type": "turn_end", "message": message_to_ts(event.message),
                "toolResults": [message_to_ts(item) for item in event.tool_results]}
    if kind in ("message_start", "message_end"):
        return {"type": kind, "message": message_to_ts(event.message)}
    if kind == "tool_execution_start":
        return {"type": kind, "toolCallId": event.tool_call_id, "toolName": event.tool_name, "args": event.args}
    if kind == "tool_execution_update":
        return {"type": kind, "toolCallId": event.tool_call_id, "toolName": event.tool_name,
                "args": event.args, "partialResult": tool_result_to_ts(event.partial_result)}
    if kind == "tool_execution_end":
        return {"type": kind, "toolCallId": event.tool_call_id, "toolName": event.tool_name,
                "result": tool_result_to_ts(event.result), "isError": event.is_error}
    if kind == "agent_end":
        # The Python loop has no retry engine, so willRetry is always false.
        return {"type": "agent_end", "messages": [message_to_ts(item) for item in event.messages],
                "willRetry": False}
    raise ValueError(f"Unknown agent event: {kind!r}")


def entry_to_ts(entry: SessionEntry) -> dict:
    """One session entry in the TS SessionEntry JSON shape."""
    common: Dict[str, Any] = {"id": entry.id, "parentId": entry.parent_id}
    if entry.timestamp:
        # TS entries always carry a timestamp; legacy Python records may not.
        common["timestamp"] = entry.timestamp
    if entry.message is not None and entry.custom is None:
        return {"type": "message", **common, "message": message_to_ts(entry.message)}
    if entry.message is not None and entry.custom is not None:
        result = {"type": "custom_message", **common,
                  "customType": entry.custom["kind"],
                  "content": content_blocks_to_ts(entry.message.content),
                  "display": entry.custom["data"].get("display", True)}
        details = entry.custom["data"].get("details")
        if details is not None:
            result["details"] = details
        return result
    if entry.custom is not None:
        kind = entry.custom["kind"]
        data = entry.custom.get("data", {})
        if kind.startswith("ts_") and isinstance(data, dict):
            # Opaque TS entries keep their original record shape.
            record = dict(data)
            record.setdefault("id", entry.id)
            record.setdefault("parentId", entry.parent_id)
            return record
        mapped = custom_to_ts_entry(kind, data if isinstance(data, dict) else {})
        if mapped is not None:
            return {**common, **mapped}
        return {"type": "custom", **common, "customType": kind, "data": data}
    return {"type": "custom", **common, "customType": "custom", "data": {}}


def tree_nodes_to_ts(entries: List[SessionEntry]) -> List[dict]:
    """Build TS SessionTreeNode[] roots from the append-order entry list.

    Orphans become roots; children stay in append (oldest-first) order. The
    Python port has no label entries, so ``label``/``labelTimestamp`` are never
    emitted.
    """
    nodes: Dict[str, dict] = {}
    roots: List[dict] = []
    for entry in entries:
        node = {"entry": entry_to_ts(entry), "children": []}
        nodes[entry.id] = node
        parent = nodes.get(entry.parent_id)
        if parent is not None:
            parent["children"].append(node)
        else:
            roots.append(node)
    return roots


def compaction_to_ts(result: dict) -> dict:
    """compact_agent output as TS CompactionResult.

    ``firstKeptEntryId`` is omitted: the Python compactor's checkpoint is a
    message count plus digest, not an entry id.
    """
    data: Dict[str, Any] = {"summary": result.get("summary", ""),
                            "tokensBefore": result.get("tokens_before", 0)}
    if "tokens_after" in result:
        data["estimatedTokensAfter"] = result["tokens_after"]
    if result.get("usage") is not None:
        data["usage"] = usage_to_ts(result["usage"])
    return data


def stats_to_ts(stats: dict, session_id: str) -> dict:
    """usage.session_stats output as the TS SessionStats JSON shape."""
    tokens = stats.get("tokens", {})
    result: Dict[str, Any] = {}
    session_file = stats.get("session_file")
    if session_file:
        result["sessionFile"] = session_file
    result["sessionId"] = session_id
    result["userMessages"] = stats.get("user_messages", 0)
    result["assistantMessages"] = stats.get("assistant_messages", 0)
    result["toolCalls"] = stats.get("tool_calls", 0)
    result["toolResults"] = stats.get("tool_results", 0)
    result["totalMessages"] = stats.get("total_messages", 0)
    result["tokens"] = {
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "cacheRead": tokens.get("cache_read", 0),
        "cacheWrite": tokens.get("cache_write", 0),
        "total": tokens.get("total", 0),
    }
    result["cost"] = stats.get("cost", 0.0)
    context = stats.get("context_usage")
    if context is not None:
        result["contextUsage"] = {
            "tokens": context.get("tokens"),
            "contextWindow": context.get("context_window"),
            "percent": context.get("percent"),
        }
    return result
