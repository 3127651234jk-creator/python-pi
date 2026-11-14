"""Agent runtime types.

Python port of ``packages/agent/src/types.ts``: tool definitions, tool
results, agent context, events, and the loop configuration callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from ..ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    Tool,
    ToolResultMessage,
    ThinkingLevel,
    Usage,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""

    content: List[Union[TextContent, ImageContent]]
    details: Any = None
    structured_content: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None
    is_error: bool = False
    terminate: bool = False


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass
class AgentTool(Tool):
    """Executable tool: declaration plus an async ``execute`` function."""

    label: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    execute: Optional[Callable[..., Awaitable[AgentToolResult]]] = None
    prepare_arguments: Optional[Callable[[Any], Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    execution_mode: Optional[Literal["sequential", "parallel"]] = None

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.name)


@dataclass
class AgentContext:
    messages: List[Any]
    tools: Optional[List[AgentTool]] = None


# ---------------------------------------------------------------------------
# Loop configuration callbacks
# ---------------------------------------------------------------------------


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: Optional[str] = None
    terminate: bool = False


@dataclass
class AfterToolCallResult:
    content: Optional[List[Union[TextContent, ImageContent]]] = None
    details: Any = None
    structured_content: Any = None
    is_error: Optional[bool] = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    args: Any
    context: AgentContext
    parent_tool_call_id: Optional[str] = None


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext
    parent_tool_call_id: Optional[str] = None


@dataclass
class AgentTurnContext:
    message: AssistantMessage
    tool_results: List[ToolResultMessage]
    context: AgentContext
    new_messages: List[Any]


@dataclass
class AgentTurnDecision:
    action: Literal["continue", "end"]


@dataclass
class AgentRequestContext:
    context: AgentContext
    model: Any
    thinking_level: ThinkingLevel


@dataclass
class AgentRequestUpdate:
    context: Optional[AgentContext] = None
    model: Any = None
    thinking_level: Optional[ThinkingLevel] = None


@dataclass
class AgentLoopTurnUpdate:
    context: Optional[AgentContext] = None
    messages: Optional[List[Any]] = None
    model: Any = None
    thinking_level: Optional[ThinkingLevel] = None


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop (port of AgentLoopConfig)."""

    model: Any
    convert_to_llm: Callable[[List[Any]], List[Any]]
    transform_context: Optional[Callable[..., Any]] = None
    get_api_key: Optional[Callable[[str], Optional[str]]] = None
    finish_turn: Optional[Callable[..., Any]] = None
    prepare_request: Optional[Callable[..., Any]] = None
    prepare_next_turn: Optional[Callable[..., Any]] = None
    get_steering_messages: Optional[Callable[[], Awaitable[List[Any]]]] = None
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[Any]]]] = None
    tool_execution: Literal["sequential", "parallel"] = "parallel"
    before_tool_call: Optional[Callable[..., Any]] = None
    after_tool_call: Optional[Callable[..., Any]] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None
    telemetry_context: Any = None
    # Prompt-caching breakpoint TTL ("5m"/"1h"); None disables cache markers.
    cache_retention: Optional[str] = "5m"
    # Session-scoped nested-call runner; the loop creates one per run when unset.
    nested_tool_call_runner: Any = None
    # Returns True for a failed assistant message that an auto-retry will
    # supersede; such attempts are flagged before message_end so subscribers
    # persist them as retried (raw history, omitted from model projections).
    should_omit_error: Optional[Callable[[Any], bool]] = None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class AgentStartEvent:
    type: str = field(default="agent_start", init=False)


@dataclass
class AgentEndEvent:
    type: str = field(default="agent_end", init=False)
    messages: List[Any] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: str = field(default="turn_start", init=False)


@dataclass
class TurnEndEvent:
    type: str = field(default="turn_end", init=False)
    message: Any = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: str = field(default="message_start", init=False)
    message: Any = None


@dataclass
class MessageUpdateEvent:
    type: str = field(default="message_update", init=False)
    message: Any = None
    assistant_message_event: Any = None


@dataclass
class MessageEndEvent:
    type: str = field(default="message_end", init=False)
    message: Any = None


@dataclass
class ToolExecutionStartEvent:
    type: str = field(default="tool_execution_start", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    parent_tool_call_id: Optional[str] = None


@dataclass
class ToolExecutionUpdateEvent:
    type: str = field(default="tool_execution_update", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None
    parent_tool_call_id: Optional[str] = None


@dataclass
class ToolExecutionEndEvent:
    type: str = field(default="tool_execution_end", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False
    parent_tool_call_id: Optional[str] = None


@dataclass
class AutoRetryStartEvent:
    """Auto-retry of a failed assistant turn is about to back off (1-indexed attempt)."""

    type: str = field(default="auto_retry_start", init=False)
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""


@dataclass
class AutoRetryEndEvent:
    """The auto-retry cycle ended: success, or exhausted/cancelled with the reason."""

    type: str = field(default="auto_retry_end", init=False)
    success: bool = False
    attempt: int = 0
    final_error: Optional[str] = None


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    AutoRetryStartEvent,
    AutoRetryEndEvent,
]
