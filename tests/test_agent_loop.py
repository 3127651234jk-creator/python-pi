"""Agent loop tests driven by the faux provider (no network)."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.agent import Agent
from pi.agent.agent_loop import (
    declare_tool_changes,
    default_convert_to_llm,
    execute_tool_calls_parallel,
    execute_tool_calls_sequential,
    run_agent_loop,
)
from pi.agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    BeforeToolCallResult,
    MessageEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi.ai.models import Model
from pi.ai.providers import faux
from pi.ai.providers.base import make_partial
from pi.ai.retry import RetryPolicy
from pi.ai.types import (
    AssistantMessage,
    DoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi.coding_agent.session import SessionManager
from pi.util import AbortSignal


@pytest.fixture(autouse=True)
def faux_script_env():
    yield
    os.environ.pop("FAUX_SCRIPT", None)
    faux.reset_script_index()


def use_script(script):
    os.environ["FAUX_SCRIPT"] = json.dumps(script)
    faux.reset_script_index()


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


def make_tool(name: str = "echo", mode: str = None) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=str(params.get("text", "ok")))], details={"echoed": params})

    return AgentTool(
        name=name,
        label=name,
        description="echo a value",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
        execution_mode=mode,
    )


def test_tool_call_roundtrip():
    """A toolUse turn followed by a text turn completes with a toolResult."""

    async def scenario():
        use_script(
            [
                {"text": "", "toolCalls": [{"id": "t1", "name": "echo", "arguments": {"text": "hi"}}]},
                {"text": "done"},
            ]
        )

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        events = []
        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[make_tool()]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: events.append(e) or asyncio.sleep(0),
            stream_fn=stream_fn,
        )

        roles = [getattr(m, "role", None) for m in messages]
        assert roles == ["system", "user", "assistant", "toolResult", "assistant"]
        assert isinstance(messages[3], ToolResultMessage)
        assert messages[3].content[0].text == "hi"
        assert messages[4].content[0].text == "done"

        event_types = [type(e).__name__ for e in events]
        assert "ToolExecutionStartEvent" in event_types
        assert "ToolExecutionEndEvent" in event_types
        assert event_types[-1] == "AgentEndEvent"

    asyncio.run(scenario())


def test_undeclared_tool_is_not_declared_to_the_model():
    """--tools keeps MCP tools registered without declaring them (04b97ef00)."""

    hidden = make_tool("mcp__dev__echo")
    hidden.declared = False
    messages = declare_tool_changes(
        AgentContext(messages=[], tools=[make_tool("read"), hidden]),
        [UserMessage(content="go", timestamp=0)],
    )
    system = next(message for message in messages if getattr(message, "role", None) == "system")
    assert [tool.name for tool in (system.tools_added or [])] == ["read"]


def test_undeclared_tool_is_removed_from_a_resumed_transcript():
    """A declaration restored from a session is removed when the tool is now undeclared."""

    hidden = make_tool("mcp__dev__echo")
    hidden.declared = False
    visible = make_tool("read")
    declared = [Tool(name="read", description=visible.description, parameters=visible.parameters),
                Tool(name="mcp__dev__echo", description=hidden.description,
                     parameters=hidden.parameters)]
    messages = declare_tool_changes(
        AgentContext(messages=[SystemMessage(content="", timestamp=0, tools_added=declared)],
                     tools=[visible, hidden]),
        [UserMessage(content="go", timestamp=0)],
    )
    removed = [ref.name for message in messages
               for ref in (getattr(message, "tools_removed", None) or [])]
    assert removed == ["mcp__dev__echo"]


def test_undeclared_tool_rejects_top_level_calls():
    async def scenario():
        use_script([
            {"text": "", "toolCalls": [{"id": "t1", "name": "mcp__dev__echo",
                                       "arguments": {"text": "hi"}}]},
            {"text": "done"},
        ])
        hidden = make_tool("mcp__dev__echo")
        hidden.declared = False

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[hidden]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        result = next(message for message in messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "not found" in result.content[0].text

    asyncio.run(scenario())


def test_undeclared_tool_is_reachable_from_a_running_tool():
    """Codemode-style nested calls reach the tools the model cannot see."""

    from pi.coding_agent.nested_tool_calls import execute_tool

    async def scenario():
        use_script([
            {"text": "", "toolCalls": [{"id": "t1", "name": "outer", "arguments": {"text": "hi"}}]},
            {"text": "done"},
        ])
        hidden = make_tool("mcp__dev__echo")
        hidden.declared = False

        async def execute(tool_call_id, params, signal=None, on_update=None):
            outcome = await execute_tool("mcp__dev__echo", {"text": params["text"]}, signal)
            text = "\n".join(block.text for block in outcome.result.content
                             if isinstance(block, TextContent))
            return AgentToolResult(content=[TextContent(text=text)])

        outer = AgentTool(name="outer", description="outer", parameters={
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            execute=execute)

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[outer, hidden]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        result = next(message for message in messages if isinstance(message, ToolResultMessage))
        assert not result.is_error and result.content[0].text == "hi"

    asyncio.run(scenario())


def test_unknown_wire_event_preserves_partial_and_completes():
    async def scenario():
        partial = make_partial(make_model(), 0)
        partial.content = [TextContent(text="in progress")]
        unknown = {"type": "future_event", "partial": partial, "extra": "payload"}

        async def stream_fn(model, context, options=None):
            yield StreamStartEvent(partial=partial)
            yield unknown
            final = make_partial(model, 0)
            final.content = [TextContent(text="complete")]
            final.stop_reason = "stop"
            yield DoneEvent(reason="stop", message=final)

        events = []
        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: events.append(e) or asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        updates = [e for e in events if getattr(e, "type", None) == "message_update"]
        assert len(updates) == 1
        assert updates[0].assistant_message_event is unknown
        assert updates[0].message.content[0].text == "in progress"
        assert messages[-1].content[0].text == "complete"
        assert events[-1].type == "agent_end"

    asyncio.run(scenario())


def test_unknown_tool_reports_error():
    async def scenario():
        use_script(
            [
                {"toolCalls": [{"id": "t1", "name": "missing", "arguments": {}}]},
                {"text": "ok"},
            ]
        )

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        # No executable tools and no declared tools: no system message is inserted.
        result = messages[2]
        assert result.is_error
        assert "not found" in result.content[0].text

    asyncio.run(scenario())


def test_invalid_arguments_report_error():
    async def scenario():
        use_script(
            [
                {"toolCalls": [{"id": "t1", "name": "echo", "arguments": {"wrong": 1}}]},
                {"text": "ok"},
            ]
        )

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[make_tool()]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        result = messages[3]
        assert result.is_error
        assert "missing required property 'text'" in result.content[0].text

    asyncio.run(scenario())


def test_before_tool_call_block():
    async def scenario():
        use_script(
            [
                {"toolCalls": [{"id": "t1", "name": "echo", "arguments": {"text": "x"}}]},
                {"text": "ok"},
            ]
        )

        async def stream_fn(model, context, options=None):
            async for event in faux.stream_faux(model, context, options):
                yield event

        blocked = []

        async def before(ctx, signal=None):
            blocked.append(ctx.tool_name)
            return BeforeToolCallResult(block=True, reason="not allowed")

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[make_tool()]),
            config=AgentLoopConfig(
                model=make_model(),
                convert_to_llm=default_convert_to_llm,
                before_tool_call=before,
            ),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        result = messages[3]
        assert result.is_error
        assert result.content[0].text == "not allowed"
        assert blocked == ["echo"]

    asyncio.run(scenario())


def test_agent_prompt_and_followup():
    async def scenario():
        use_script([{"text": "answer one"}, {"text": "answer two"}, {"text": "answer three"}])
        agent = Agent(model=make_model(), system_prompt="You are a test agent.")

        first = await agent.prompt("hello")
        assert [getattr(m, "role", None) for m in first] == ["user", "assistant"]
        assert first[1].content[0].text == "answer one"

        # A queued follow-up makes the loop continue after it would stop.
        agent.follow_up("and more")
        second = await agent.prompt("again")
        assistant_texts = [
            m.content[0].text for m in second if getattr(m, "role", None) == "assistant"
        ]
        assert assistant_texts == ["answer two", "answer three"]

        # Transcript accumulates across runs.
        roles = [getattr(m, "role", None) for m in agent.messages]
        assert roles.count("assistant") == 3

    asyncio.run(scenario())


def test_truncated_message_fails_tool_calls():
    async def scenario():
        use_script([{"text": "recovered"}])
        calls = {"count": 0}

        async def stream_fn(model, context, options=None):
            # First request: a "length" stop carrying tool calls with truncated
            # arguments. Later requests: a clean turn so the loop can finish.
            import time

            from pi.ai.providers.base import make_partial
            from pi.ai.types import (
                DoneEvent,
                StreamStartEvent,
                ToolCall,
                ToolCallDeltaEvent,
                ToolCallEndEvent,
                ToolCallStartEvent,
                Usage,
            )

            calls["count"] += 1
            if calls["count"] > 1:
                async for event in faux.stream_faux(model, context, options):
                    yield event
                return

            partial = make_partial(model, 0)
            yield StreamStartEvent(partial=partial)
            tool_call = ToolCall(id="t1", name="echo", arguments={"text": "trunc"})
            partial.content.append(tool_call)
            yield ToolCallStartEvent(content_index=0, partial=partial)
            yield ToolCallDeltaEvent(content_index=0, delta="{}", partial=partial)
            yield ToolCallEndEvent(content_index=0, tool_call=tool_call, partial=partial)
            done = make_partial(model, 0)
            done.content = list(partial.content)
            done.stop_reason = "length"
            done.usage = Usage()
            yield DoneEvent(reason="length", message=done)

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[make_tool()]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        result = messages[3]
        assert result.is_error
        assert "output token limit" in result.content[0].text
        assert calls["count"] == 2

    asyncio.run(scenario())


def test_hosted_tool_calls_are_not_executed():
    """Provider-executed (hosted) tool calls replay via raw_item: no execution,
    no tool result, and the loop continues to the next turn."""

    async def scenario():
        import time

        from pi.ai.providers.base import make_partial
        from pi.ai.types import DoneEvent, StreamStartEvent, Usage

        raw_item = {"type": "web_search_call", "id": "ws_1", "status": "completed",
                    "action": {"type": "search", "query": "q"}}

        async def stream_fn(model, context, options=None):
            partial = make_partial(model, 0)
            yield StreamStartEvent(partial=partial)
            done = make_partial(model, 0)
            done.content = [
                ToolCall(id="ws_1", name="web_search", arguments={}, raw_item=raw_item),
                TextContent(text="answer"),
            ]
            done.stop_reason = "stop"
            done.usage = Usage()
            yield DoneEvent(reason="stop", message=done)

        executed = []
        tools = [AgentTool(name="web_search", label="web_search", description="hosted",
                           parameters={"type": "object"},
                           execute=lambda *args, **kwargs: executed.append(args))]

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=tools),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        assert executed == []
        assert [getattr(m, "role", None) for m in messages] == ["system", "user", "assistant"]
        assert messages[2].content[0].raw_item == raw_item

    asyncio.run(scenario())


def test_hosted_tool_declaration_preserved_in_changes():
    """Hosted tools registered on the context re-declare with their payload."""

    async def scenario():
        import time

        from pi.ai.providers.base import make_partial
        from pi.ai.types import DoneEvent, StreamStartEvent, Tool, Usage

        async def stream_fn(model, context, options=None):
            partial = make_partial(model, 0)
            yield StreamStartEvent(partial=partial)
            done = make_partial(model, 0)
            done.content = [TextContent(text="ok")]
            done.stop_reason = "stop"
            done.usage = Usage()
            yield DoneEvent(reason="stop", message=done)

        captured = {}

        async def run(model, context, options=None):
            captured["context"] = context
            async for event in stream_fn(model, context, options):
                yield event

        tools = [AgentTool(name="web_search", label="web_search", description="hosted",
                           parameters={"type": "object"},
                           hosted={"type": "web_search", "search_context_size": "medium"})]
        await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=tools),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=run,
        )
        declared = [t for m in captured["context"].messages if getattr(m, "role", None) == "system"
                    for t in (m.tools_added or [])]
        assert [t.hosted for t in declared] == [{"type": "web_search", "search_context_size": "medium"}]

    asyncio.run(scenario())


def _abort_batch_fixture():
    """Three calls on one assistant message; the first one aborts the signal."""
    tool_calls = [ToolCall(id=f"t{i}", name="echo", arguments={"text": "x"}) for i in (1, 2, 3)]
    assistant = make_partial(make_model(), 0)
    assistant.content = tool_calls
    assistant.stop_reason = "toolUse"
    return tool_calls, assistant


def _assert_batch_answers_every_call(batch, events):
    results = {message.tool_call_id: message for message in batch["messages"]}
    assert set(results) == {"t1", "t2", "t3"}
    assert results["t1"].content[0].text == "ran"
    for call_id in ("t2", "t3"):
        assert results[call_id].is_error
        assert results[call_id].content[0].text == "Operation aborted"
    starts = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
    ends = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert starts == ["t1", "t2", "t3"]
    assert ends == ["t1", "t2", "t3"]


def test_abort_answers_remaining_tool_calls_sequential():
    """An aborted serial batch still emits a toolResult for every toolUse block."""

    async def scenario():
        signal = AbortSignal()
        tool_calls, assistant = _abort_batch_fixture()

        async def execute(tool_call_id, params, sig=None, on_update=None):
            if tool_call_id == "t1":
                sig.abort()
            return AgentToolResult(content=[TextContent(text="ran")], details={})

        tool = make_tool()
        tool.execute = execute
        events = []
        batch = await execute_tool_calls_sequential(
            AgentContext(messages=[], tools=[tool]),
            assistant,
            tool_calls,
            AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal,
            lambda e: events.append(e) or asyncio.sleep(0),
        )
        _assert_batch_answers_every_call(batch, events)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Agent auto-retry
# ---------------------------------------------------------------------------


def test_agent_auto_retries_transient_error():
    async def scenario():
        use_script([{"error": "terminated"}, {"text": "recovered"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert [getattr(m, "role", None) for m in messages] == ["user", "assistant", "assistant"]
        assert messages[-1].content[0].text == "recovered"
        # The superseded attempt stays in raw history (so memory and the session
        # log agree) but is omitted from the model projection.
        assert messages[1].stop_reason == "error"
        assert messages[1].retried is True
        assert agent.messages[-1].content[0].text == "recovered"
        assert [m.role for m in default_convert_to_llm(agent.messages)] == ["user", "assistant"]
        starts = [e for e in events if isinstance(e, AutoRetryStartEvent)]
        ends = [e for e in events if isinstance(e, AutoRetryEndEvent)]
        assert [(s.attempt, s.max_attempts, s.delay_ms) for s in starts] == [(1, 3, 0)]
        assert starts[0].error_message == "terminated"
        assert [(e.success, e.attempt) for e in ends] == [(True, 1)]

    asyncio.run(scenario())


def test_abort_answers_remaining_tool_calls_parallel():
    """An abort during preparation still answers the calls that never started."""

    async def scenario():
        signal = AbortSignal()
        tool_calls, assistant = _abort_batch_fixture()

        async def before(ctx, sig):
            if ctx.tool_call_id == "t1":
                sig.abort()
            return None

        events = []
        batch = await execute_tool_calls_parallel(
            AgentContext(messages=[], tools=[make_tool()]),
            assistant,
            tool_calls,
            AgentLoopConfig(
                model=make_model(),
                convert_to_llm=default_convert_to_llm,
                before_tool_call=before,
            ),
            signal,
            lambda e: events.append(e) or asyncio.sleep(0),
        )
        results = {message.tool_call_id: message for message in batch["messages"]}
        assert set(results) == {"t1", "t2", "t3"}
        for call_id in ("t1", "t2", "t3"):
            assert results[call_id].is_error
            assert results[call_id].content[0].text == "Operation aborted"
        starts = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
        ends = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
        assert starts == ["t1", "t2", "t3"]
        assert ends == ["t1", "t2", "t3"]

    asyncio.run(scenario())


def test_retried_attempt_round_trips_through_session(tmp_path):
    """A retried-away attempt is persisted and re-omitted after a session reload."""

    async def scenario():
        use_script([{"error": "terminated"}, {"text": "recovered"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0))
        session = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
        session.start_new()

        async def record(event):
            if isinstance(event, MessageEndEvent):
                session.append(event.message)

        agent.subscribe(record)
        await agent.prompt("hello")

        loaded = session.load(session.session_file)
        assert [getattr(m, "role", None) for m in loaded] == ["user", "assistant", "assistant"]
        assert loaded[1].retried is True
        assert [m.role for m in default_convert_to_llm(loaded)] == ["user", "assistant"]
        assert loaded[-1].content[0].text == "recovered"

    asyncio.run(scenario())


def test_stream_events_survives_auto_retry():
    """The event stream only ends when the prompt does, not on the failed attempt."""

    async def scenario():
        use_script([{"error": "terminated"}, {"text": "recovered"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=100))
        types = []
        async for event in agent.stream_events("hello"):
            types.append(event.type)
        assert types.count("agent_end") == 2
        assert "auto_retry_start" in types
        assert "auto_retry_end" in types
        assert agent.messages[-1].content[0].text == "recovered"

    asyncio.run(scenario())


def test_agent_auto_retry_does_not_retry_non_retryable_error():
    async def scenario():
        use_script([{"error": "insufficient_quota"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert messages[-1].stop_reason == "error"
        assert messages[-1].error_message == "insufficient_quota"
        # A deterministic error is neither discarded nor omitted.
        assert messages[-1].retried is False
        assert agent.messages[-1].stop_reason == "error"
        assert len(default_convert_to_llm(agent.messages)) == 2
        assert not [e for e in events if isinstance(e, (AutoRetryStartEvent, AutoRetryEndEvent))]

    asyncio.run(scenario())


def test_abort_during_tool_batch_does_not_request_another_model_turn():
    async def scenario():
        signal = AbortSignal()
        provider_calls = 0

        async def stream_fn(model, context, options=None):
            nonlocal provider_calls
            provider_calls += 1
            partial = make_partial(model, 0)
            yield StreamStartEvent(partial=partial)
            message = make_partial(model, 0)
            message.content = [ToolCall(id="t1", name="echo", arguments={"text": "x"})]
            message.stop_reason = "toolUse"
            yield DoneEvent(reason="toolUse", message=message)

        async def execute(tool_call_id, params, sig=None, on_update=None):
            sig.abort()
            return AgentToolResult(content=[TextContent(text="ran")], details={})

        tool = make_tool()
        tool.execute = execute
        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[tool]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=signal,
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )

        assert provider_calls == 1
        assert signal.aborted
        assert [getattr(message, "role", None) for message in messages] == [
            "system", "user", "assistant", "toolResult",
        ]

    asyncio.run(scenario())


def test_agent_auto_retry_skips_context_overflow():
    async def scenario():
        use_script([{"error": "context_length_exceeded"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert messages[-1].stop_reason == "error"
        assert messages[-1].retried is False
        assert not [e for e in events if isinstance(e, (AutoRetryStartEvent, AutoRetryEndEvent))]

    asyncio.run(scenario())


def test_agent_auto_retry_gives_up_after_max_retries():
    async def scenario():
        use_script([{"error": "terminated"}] * 4)
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0, max_retries=2))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert messages[-1].stop_reason == "error"
        # The two superseded attempts are marked; the final error is not.
        assert [m.retried for m in messages[1:]] == [True, True, False]
        starts = [e.attempt for e in events if isinstance(e, AutoRetryStartEvent)]
        assert starts == [1, 2]
        end = next(e for e in events if isinstance(e, AutoRetryEndEvent))
        assert end.success is False
        assert end.attempt == 2
        assert end.final_error == "terminated"

    asyncio.run(scenario())


def test_agent_auto_retry_resets_budget_after_success():
    async def scenario():
        use_script([
            {"error": "terminated"},
            {"toolCalls": [{"id": "t1", "name": "echo", "arguments": {"text": "x"}}]},
            {"error": "terminated"},
            {"text": "done"},
        ])
        agent = Agent(model=make_model(), tools=[make_tool()], retry=RetryPolicy(base_delay_ms=0))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert messages[-1].content[0].text == "done"
        # Both superseded attempts are omitted from the projection; the tool
        # round and the final answer remain.
        assert [m.role for m in default_convert_to_llm(messages)] == [
            "system", "user", "assistant", "toolResult", "assistant",
        ]
        # The budget reset after the successful tool-call turn: both retries are attempt 1.
        starts = [e.attempt for e in events if isinstance(e, AutoRetryStartEvent)]
        assert starts == [1, 1]
        ends = [(e.success, e.attempt) for e in events if isinstance(e, AutoRetryEndEvent)]
        # TS emits auto_retry_end on every non-error assistant message after a retry:
        # the tool-call turn and the final answer each end a retry cycle.
        assert ends == [(True, 1), (True, 1)]

    asyncio.run(scenario())


def test_aborted_stream_answers_streamed_tool_calls():
    """A request aborted after a tool_use block was streamed still gets a result."""

    async def scenario():
        from pi.ai.types import StreamErrorEvent, StreamStartEvent

        def aborted_message(model):
            partial = make_partial(model, 0)
            partial.content = [ToolCall(id="t1", name="echo", arguments={})]
            partial.stop_reason = "aborted"
            return partial

        async def stream_fn(model, context, options=None):
            yield StreamStartEvent(partial=aborted_message(model))
            yield StreamErrorEvent(reason="aborted", error=aborted_message(model))

        messages = await run_agent_loop(
            prompts=[UserMessage(content="go", timestamp=0)],
            context=AgentContext(messages=[], tools=[make_tool()]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=AbortSignal(),
            emit=lambda e: asyncio.sleep(0),
            stream_fn=stream_fn,
        )
        roles = [getattr(m, "role", None) for m in messages]
        assert roles == ["system", "user", "assistant", "toolResult"]
        result = messages[-1]
        assert result.tool_call_id == "t1"
        assert result.is_error
        assert "aborted" in result.content[0].text.lower()

    asyncio.run(scenario())


def test_agent_auto_retry_aborts_during_backoff():
    async def scenario():
        use_script([{"error": "terminated"}, {"text": "unused"}])
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=50_000))
        events = []
        agent.subscribe(events.append)
        task = asyncio.ensure_future(agent.prompt("hello"))
        while not any(isinstance(e, AutoRetryStartEvent) for e in events):
            await asyncio.sleep(0.01)
        agent.abort()
        messages = await task
        # The failed attempt is retained (marked) and the cancellation ends the run.
        assert [getattr(m, "role", None) for m in messages] == ["user", "assistant"]
        assert messages[1].retried is True
        end = next(e for e in events if isinstance(e, AutoRetryEndEvent))
        assert end.success is False
        assert end.attempt == 1
        assert end.final_error == "Retry cancelled"

    asyncio.run(scenario())


def test_agent_auto_retry_reports_aborted_retry_as_failed(monkeypatch):
    """An aborted retried attempt is not reported as a successful retry."""

    calls = {"n": 0}

    async def stream(model, context, options):
        calls["n"] += 1
        if calls["n"] == 1:
            error = AssistantMessage(
                content=[], api="faux", provider="faux", model="faux", usage=Usage(),
                stop_reason="error", timestamp=0, error_message="terminated",
            )
            yield StreamErrorEvent(reason="error", error=error)
        else:
            aborted = AssistantMessage(
                content=[], api="faux", provider="faux", model="faux", usage=Usage(),
                stop_reason="aborted", timestamp=0,
            )
            yield StreamErrorEvent(reason="aborted", error=aborted)

    monkeypatch.setattr("pi.agent.agent.get_default_stream_fn", lambda: stream)

    async def scenario():
        agent = Agent(model=make_model(), retry=RetryPolicy(base_delay_ms=0))
        events = []
        agent.subscribe(events.append)
        messages = await agent.prompt("hello")
        assert messages[-1].stop_reason == "aborted"
        assert messages[1].retried is True
        ends = [e for e in events if isinstance(e, AutoRetryEndEvent)]
        assert [(e.success, e.attempt, e.final_error) for e in ends] == [(False, 1, "Retry cancelled")]

    asyncio.run(scenario())
