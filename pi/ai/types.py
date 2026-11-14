"""Message and event types for the pi.ai layer.

Python port of ``packages/ai/src/types.ts``. The message model is the same
four-role transcript: system, user, assistant, toolResult. Content blocks are
dataclasses; system messages carry the prompt and tool declarations.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

EMPTY_USAGE = {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "cache_write": 0,
    "cache_write_1h": 0,
    "total_tokens": 0,
    "cost": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "cache_write_1h": 0.0, "total": 0.0},
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0
    reasoning: Optional[int] = None
    cost: Dict[str, float] = field(
        default_factory=lambda: {
            "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "cache_write_1h": 0.0, "total": 0.0
        }
    )

    @property
    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write + self.cache_write_1h

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total_tokens": self.total_tokens,
            "cost": dict(self.cost),
        }
        if self.cache_write_1h:
            d["cache_write_1h"] = self.cache_write_1h
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def empty(cls) -> "Usage":
        return cls()


def combine_usage(first: Usage, second: Usage) -> Usage:
    """Sum of two usages, keeping optional token splits when either side reports them."""
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        cache_write_1h=first.cache_write_1h + second.cache_write_1h,
        reasoning=None if first.reasoning is None and second.reasoning is None
        else (first.reasoning or 0) + (second.reasoning or 0),
        cost={key: first.cost.get(key, 0.0) + second.cost.get(key, 0.0)
              for key in set(first.cost) | set(second.cost)},
    )


@dataclass
class TextContent:
    type: str = field(default="text", init=False)
    text: str
    text_signature: Optional[str] = None


@dataclass
class ThinkingContent:
    type: str = field(default="thinking", init=False)
    thinking: str
    thinking_signature: Optional[str] = None
    redacted: bool = False


@dataclass
class ImageContent:
    type: str = field(default="image", init=False)
    data: str  # base64
    mime_type: str


@dataclass
class AudioContent:
    type: str = field(default="audio", init=False)
    data: str  # base64
    mime_type: str


@dataclass
class VideoContent:
    type: str = field(default="video", init=False)
    data: str  # base64
    mime_type: str


@dataclass
class ToolCall:
    type: str = field(default="toolCall", init=False)
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None
    # Hosted (provider-executed) tool call: the raw provider output item,
    # replayed verbatim on subsequent requests. None for client-executed calls.
    raw_item: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    """A tool declaration sent to the model (name/description/JSON schema).

    ``hosted`` marks a provider-executed tool: the dict is passed through
    verbatim as the provider's tool payload (e.g. ``{"type": "web_search"}``
    for the Responses API); name/description are client-side labels only.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    hosted: Optional[Dict[str, Any]] = None


@dataclass
class ToolReference:
    name: str


@dataclass
class SystemMessage:
    role: str = field(default="system", init=False)
    content: Union[str, List[TextContent]]
    timestamp: int
    sections: Optional[Dict[str, Optional[str]]] = None
    tools_added: Optional[List[Tool]] = None
    tools_removed: Optional[List[ToolReference]] = None


@dataclass
class UserMessage:
    role: str = field(default="user", init=False)
    content: Union[str, List[Union[TextContent, ImageContent, AudioContent, VideoContent]]]
    timestamp: int


@dataclass
class DiagnosticErrorInfo:
    """Redacted exception information carried by a message diagnostic."""

    message: str
    name: Optional[str] = None
    stack: Optional[str] = None
    code: Optional[Union[str, int]] = None


@dataclass
class AssistantMessageDiagnostic:
    """Redacted provider/runtime diagnostic for failures and recoveries."""

    type: str
    timestamp: int
    error: Optional[DiagnosticErrorInfo] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class AssistantMessage:
    role: str = field(default="assistant", init=False)
    content: List[Union[TextContent, ThinkingContent, ToolCall]]
    api: str
    provider: str
    model: str
    usage: Usage
    stop_reason: StopReason
    timestamp: int
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    thinking_level: Optional[ThinkingLevel] = None
    error_message: Optional[str] = None
    raw_stop_reason: Optional[str] = None
    end_turn: Optional[bool] = None
    provider_thinking_level: Optional[str] = None
    diagnostics: Optional[List[AssistantMessageDiagnostic]] = None
    # A failed attempt superseded by an auto-retry: kept in raw history and
    # session logs, omitted from model projections.
    retried: bool = False


def format_thrown_value(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value) or value.__class__.__name__
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    """Extract redacted exception information (port of extractDiagnosticError)."""
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    code = getattr(error, "code", None)
    stack = "".join(traceback.format_exception(type(error), error, error.__traceback__)) if error.__traceback__ else None
    return DiagnosticErrorInfo(
        name=error.__class__.__name__,
        message=str(error) or error.__class__.__name__,
        stack=stack,
        code=code if isinstance(code, (str, int)) and not isinstance(code, bool) else None,
    )


def create_assistant_message_diagnostic(
    diagnostic_type: str, error: Any = None, details: Optional[Dict[str, Any]] = None
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=diagnostic_type,
        timestamp=int(time.time() * 1000),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: AssistantMessage, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]


@dataclass
class NestedToolCallRecord:
    """A tool call that another tool made while it ran, for example from a codemode script."""

    id: str
    parent_tool_call_id: Optional[str]
    name: str
    # Omitted when over the size limits; arguments_bytes then gives their size.
    arguments: Optional[Dict[str, Any]] = None
    # UTF-8 size of the arguments as compact JSON, set when arguments is omitted.
    arguments_bytes: Optional[int] = None
    # unfinished: the call was still running when the calling tool finished.
    status: str = "unfinished"
    duration_ms: Optional[int] = None
    # Error text, truncated.
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "parent_tool_call_id": self.parent_tool_call_id,
            "name": self.name,
            "status": self.status,
        }
        if self.arguments is not None:
            data["arguments"] = self.arguments
        if self.arguments_bytes is not None:
            data["arguments_bytes"] = self.arguments_bytes
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.error is not None:
            data["error"] = self.error
        return data


@dataclass
class NestedToolCalls:
    """Bounded record of the nested calls a tool made. Results are not recorded."""

    calls: List[NestedToolCallRecord] = field(default_factory=list)
    # False when calls were dropped, arguments omitted, or calls had not finished.
    complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"calls": [call.to_dict() for call in self.calls], "complete": self.complete}


@dataclass
class ToolResultMessage:
    role: str = field(default="toolResult", init=False)
    tool_call_id: str
    tool_name: str
    content: List[Union[TextContent, ImageContent]]
    is_error: bool
    timestamp: int
    details: Any = None
    usage: Optional[Usage] = None
    # Calls this tool made to other tools. Kept for the session record; not sent to the model.
    nested_calls: Optional[NestedToolCalls] = None


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]
AssistantContent = Union[TextContent, ThinkingContent, ToolCall]
ToolResultContent = Union[TextContent, ImageContent]
UserContent = Union[TextContent, ImageContent, AudioContent, VideoContent]


@dataclass
class Context:
    """Request input accepted by the public stream entry points."""

    system_prompt: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    tools: Optional[List[Tool]] = None


# ---------------------------------------------------------------------------
# Image generation (port of ImagesContext / ImagesOptions / AssistantImages)
# ---------------------------------------------------------------------------


@dataclass
class ImagesContext:
    """Input for one image generation request."""

    input: List[Union[TextContent, ImageContent]] = field(default_factory=list)


@dataclass
class ImagesOptions:
    """Options for image generation calls."""

    signal: Optional[Any] = None  # util.AbortSignal
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_ms: Optional[float] = None


@dataclass
class AssistantImages:
    """Result of one image generation request (port of AssistantImages)."""

    api: str
    provider: str
    model: str
    output: List[Union[TextContent, ImageContent]] = field(default_factory=list)
    stop_reason: str = "stop"
    timestamp: int = 0
    response_id: Optional[str] = None
    usage: Optional[Usage] = None
    error_message: Optional[str] = None


@dataclass
class TranscriptContext:
    """Normalized context: prompt and tools live in the leading system message."""

    messages: List[Message]


# ---------------------------------------------------------------------------
# Assistant message event protocol (port of AssistantMessageEvent)
# ---------------------------------------------------------------------------


@dataclass
class StreamStartEvent:
    type: str = field(default="start", init=False)
    partial: AssistantMessage = None  # type: ignore[assignment]


@dataclass
class TextStartEvent:
    type: str = field(default="text_start", init=False)
    content_index: int
    partial: AssistantMessage


@dataclass
class TextDeltaEvent:
    type: str = field(default="text_delta", init=False)
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class TextEndEvent:
    type: str = field(default="text_end", init=False)
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass
class ThinkingStartEvent:
    type: str = field(default="thinking_start", init=False)
    content_index: int
    partial: AssistantMessage


@dataclass
class ThinkingDeltaEvent:
    type: str = field(default="thinking_delta", init=False)
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class ThinkingEndEvent:
    type: str = field(default="thinking_end", init=False)
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass
class ToolCallStartEvent:
    type: str = field(default="toolcall_start", init=False)
    content_index: int
    partial: AssistantMessage


@dataclass
class ToolCallDeltaEvent:
    type: str = field(default="toolcall_delta", init=False)
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass
class ToolCallEndEvent:
    type: str = field(default="toolcall_end", init=False)
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


@dataclass
class DoneEvent:
    type: str = field(default="done", init=False)
    reason: StopReason
    message: AssistantMessage


@dataclass
class StreamErrorEvent:
    type: str = field(default="error", init=False)
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Union[
    StreamStartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    StreamErrorEvent,
]


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message: base content, then sections in insertion order."""
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "\n".join(block.text for block in message.content)
    parts = [text]
    parts.extend(section for section in (message.sections or {}).values() if section is not None)
    return "\n\n".join(part for part in parts if part)


def get_message_text(content: Union[str, List[Any]]) -> str:
    """Flatten user/assistant/toolResult content to plain text."""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            continue
        elif isinstance(block, ToolCall):
            parts.append(f'[tool call: {block.name}({block.arguments})]')
    return "\n".join(parts)
