"""Agent loop tests driven by the faux provider (no network)."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.agent import Agent
from pi.agent.agent_loop import default_convert_to_llm, run_agent_loop
from pi.agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    ToolExecutionEndEvent,
)
from pi.ai.models import Model
from pi.ai.providers import faux
from pi.ai.types import SystemMessage, TextContent, ToolCall, ToolResultMessage, UserMessage


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
