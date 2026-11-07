"""JSON conversion for pi-ai messages.

Shared by the coding-agent session store and the durable storage backends.
Dataclasses round-trip through plain dicts keyed by their ``role``/``type``
discriminators.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from .types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    AudioContent,
    DiagnosticErrorInfo,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Usage,
    UserMessage,
    VideoContent,
)


def to_json(value: Any) -> Any:
    """Convert pi-ai values (and containers of them) to JSON-safe data."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [to_json(item) for item in value]
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    return value


def message_from_json(data: Dict[str, Any]) -> Any:
    role = data.get("role")
    if role == "system":
        content = data.get("content", "")
        return SystemMessage(
            content=content,
            timestamp=data.get("timestamp", 0),
            sections=data.get("sections"),
            tools_added=[_tool_from_json(t) for t in (data.get("tools_added") or [])] or None,
            tools_removed=[ToolReference(**t) for t in (data.get("tools_removed") or [])] or None,
        )
    if role == "user":
        return UserMessage(content=_content_from_json(data.get("content", "")), timestamp=data.get("timestamp", 0))
    if role == "assistant":
        blocks = []
        for block in data.get("content", []):
            kind = block.get("type")
            if kind == "text":
                blocks.append(TextContent(text=block.get("text", ""), text_signature=block.get("text_signature")))
            elif kind == "thinking":
                blocks.append(
                    ThinkingContent(
                        thinking=block.get("thinking", ""),
                        thinking_signature=block.get("thinking_signature"),
                        redacted=block.get("redacted", False),
                    )
                )
            elif kind == "toolCall":
                blocks.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("arguments", {}),
                    thought_signature=block.get("thought_signature"), raw_item=block.get("raw_item")))
        usage_data = data.get("usage") or {}
        return AssistantMessage(
            content=blocks,
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage=usage_from_json(usage_data),
            stop_reason=data.get("stop_reason", "stop"),
            timestamp=data.get("timestamp", 0),
            response_model=data.get("response_model"),
            response_id=data.get("response_id"),
            thinking_level=data.get("thinking_level"),
            error_message=data.get("error_message"),
            raw_stop_reason=data.get("raw_stop_reason"),
            end_turn=data.get("end_turn"),
            provider_thinking_level=data.get("provider_thinking_level"),
            diagnostics=[_diagnostic_from_json(item) for item in (data.get("diagnostics") or [])
                         if isinstance(item, dict)] or None,
        )
    if role == "toolResult":
        blocks = [_block_from_json(block) for block in data.get("content", [])]
        usage_data = data.get("usage")
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=blocks,
            is_error=data.get("is_error", False),
            timestamp=data.get("timestamp", 0),
            details=data.get("details"),
            usage=usage_from_json(usage_data) if usage_data else None,
        )
    raise ValueError(f"Unknown message role: {role!r}")


def messages_from_json(items: List[Dict[str, Any]]) -> List[Any]:
    return [message_from_json(item) for item in items]


def _tool_from_json(data: Dict[str, Any]) -> Tool:
    return Tool(name=data["name"], description=data.get("description", ""), parameters=data.get("parameters", {}),
        hosted=data.get("hosted"))


def _diagnostic_from_json(data: Dict[str, Any]) -> AssistantMessageDiagnostic:
    error = data.get("error")
    return AssistantMessageDiagnostic(
        type=data.get("type", ""),
        timestamp=data.get("timestamp", 0),
        error=DiagnosticErrorInfo(
            message=error.get("message", ""),
            name=error.get("name"),
            stack=error.get("stack"),
            code=error.get("code"),
        ) if isinstance(error, dict) else None,
        details=data.get("details"),
    )


def usage_from_json(data: Dict[str, Any]) -> Usage:
    fields = {k: v for k, v in data.items() if k in Usage.__dataclass_fields__}
    return Usage(**fields)


_MEDIA_BLOCK_TYPES = {"image": ImageContent, "audio": AudioContent, "video": VideoContent}


def _block_from_json(block: Dict[str, Any]) -> Any:
    cls = _MEDIA_BLOCK_TYPES.get(block.get("type"))
    if cls is not None:
        return cls(data=block.get("data", ""), mime_type=block.get("mime_type", ""))
    return TextContent(text=block.get("text", ""))


def _content_from_json(content: Any) -> Any:
    if isinstance(content, str):
        return content
    return [_block_from_json(block) for block in content]
