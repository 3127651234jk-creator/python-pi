"""Message and event types for the pi.ai layer.

Python port of ``packages/ai/src/types.ts``. The message model is the same
four-role transcript: system, user, assistant, toolResult. Content blocks are
dataclasses; system messages carry the prompt and tool declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

EMPTY_USAGE = {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "cache_write": 0,
    "total_tokens": 0,
    "cost": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0},
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: Optional[int] = None
    cost: Dict[str, float] = field(
        default_factory=lambda: {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0}
    )

    @property
    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total_tokens": self.total_tokens,
            "cost": dict(self.cost),
        }
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def empty(cls) -> "Usage":
        return cls()


@dataclass
class TextContent:
    type: str = field(default="text", init=False)
    text: str


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
class ToolCall:
    type: str = field(default="toolCall", init=False)
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None


@dataclass
class Tool:
    """A tool declaration sent to the model (name/description/JSON schema)."""

    name: str
    description: str
    parameters: Dict[str, Any]


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
    content: Union[str, List[Union[TextContent, ImageContent]]]
    timestamp: int


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


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]
AssistantContent = Union[TextContent, ThinkingContent, ToolCall]
ToolResultContent = Union[TextContent, ImageContent]


@dataclass
class Context:
    """Request input accepted by the public stream entry points."""

    system_prompt: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    tools: Optional[List[Tool]] = None


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
    if message.sections:
        rendered = []
        for name, section in message.sections.items():
            if section is None:
                continue
            rendered.append(section if name == "preamble" else f"<{name}>\n{section}\n</{name}>")
        if rendered:
            text = (text + "\n\n" if text else "") + "\n\n".join(rendered)
    return text


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
