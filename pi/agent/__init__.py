"""pi.agent: stateful agent runtime and the agent loop.

Python port of ``packages/agent``.
"""

from .agent import Agent
from .agent_loop import (
    default_convert_to_llm,
    declare_tool_changes,
    run_agent_loop,
    run_agent_loop_continue,
)
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentRequestContext,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    "Agent",
    "AgentLoopConfig",
    "AgentContext",
    "AgentTool",
    "AgentToolResult",
    "AgentEvent",
    "AgentTurnContext",
    "AgentTurnDecision",
    "AgentLoopTurnUpdate",
    "AgentRequestContext",
    "AgentRequestUpdate",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "run_agent_loop",
    "run_agent_loop_continue",
    "declare_tool_changes",
    "default_convert_to_llm",
]
