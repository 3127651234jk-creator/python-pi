"""Regression tests for event-stream task ownership and failure cleanup."""

import asyncio
import json

import pytest

from pi.agent.agent import Agent
from pi.agent.types import AgentStartEvent, AgentTool, AgentToolResult
from pi.ai.models import Model
from pi.ai.providers import faux
from pi.ai.types import TextContent
from pi.util import AbortError


def make_agent():
    return Agent(model=Model(id="faux", name="Faux", api="faux", provider="faux", base_url=""))


def test_stream_events_propagates_prompt_failure(monkeypatch):
    async def scenario():
        agent = make_agent()

        async def fail(text):
            raise RuntimeError("prompt failed")

        monkeypatch.setattr(agent, "prompt", fail)
        started = asyncio.get_running_loop().time()
        with pytest.raises(RuntimeError, match="prompt failed"):
            await asyncio.wait_for(agent.stream_events("hello").__anext__(), 1)
        assert asyncio.get_running_loop().time() - started < 0.5
        assert not agent._listeners

    asyncio.run(scenario())


def test_stream_events_close_cancels_owned_prompt(monkeypatch):
    async def scenario():
        agent = make_agent()
        stopped = asyncio.Event()

        async def prompt(text):
            try:
                await agent._emit(AgentStartEvent())
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(agent, "prompt", prompt)
        stream = agent.stream_events("hello")
        await asyncio.wait_for(stream.__anext__(), 1)
        await asyncio.wait_for(stream.aclose(), 1)
        assert stopped.is_set()
        assert not agent._listeners

    asyncio.run(scenario())


def test_configuration_failure_clears_streaming_state(monkeypatch):
    async def scenario():
        agent = make_agent()

        def fail():
            raise RuntimeError("invalid configuration")

        monkeypatch.setattr(agent, "_build_config", fail)
        with pytest.raises(RuntimeError, match="invalid configuration"):
            await agent.prompt("hello")
        assert not agent.is_streaming
        assert agent._abort is None

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_task", [False, True])
def test_interrupted_compaction_keeps_prompt(monkeypatch, cancel_task):
    async def scenario():
        agent = make_agent()
        entered = asyncio.Event()

        async def interrupted_stream(model, context, options, stream):
            entered.set()
            await options.signal.wait()
            options.signal.throw_if_aborted()
            yield

        monkeypatch.setattr(agent.compactor, "stream", interrupted_stream)
        task = asyncio.create_task(agent.prompt("keep this prompt"))
        await entered.wait()
        if cancel_task:
            task.cancel()
        else:
            agent.abort()
        with pytest.raises(asyncio.CancelledError if cancel_task else AbortError):
            await task
        assert [m.content for m in agent.messages] == ["keep this prompt"]
        assert not agent.is_streaming
        assert agent._abort is None

    asyncio.run(scenario())


def test_subscriber_failure_keeps_persisted_tool_result(monkeypatch):
    async def scenario():
        faux.reset_script_index()
        monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
            {"toolCalls": [{"id": "call", "name": "echo", "arguments": {}}]},
            {"text": "recovered"},
        ]))
        agent = make_agent()

        async def execute(*args):
            return AgentToolResult(content=[TextContent(text="saved result")])

        agent.tools = [AgentTool(name="echo", description="echo", execute=execute)]
        persisted = []

        def persist(event):
            if event.type == "message_end":
                persisted.append(event.message)

        def fail(event):
            if event.type == "message_end" and event.message.role == "toolResult":
                raise RuntimeError("subscriber failed")

        agent.subscribe(persist)
        unsubscribe = agent.subscribe(fail)
        try:
            with pytest.raises(RuntimeError, match="subscriber failed"):
                await agent.prompt("run tool")
            assert agent.messages == persisted
            assert [m.role for m in agent.messages] == ["system", "user", "assistant", "toolResult"]
            assert agent.messages[-1].content[0].text == "saved result"
            assert not agent.is_streaming
            unsubscribe()
            new_messages = await agent.prompt("continue")
            assert agent.messages == persisted
            assert len(new_messages) == 2
        finally:
            faux.reset_script_index()

    asyncio.run(scenario())


def test_subscriber_failure_keeps_every_completed_parallel_tool_result(monkeypatch):
    async def scenario():
        faux.reset_script_index()
        monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
            {"toolCalls": [
                {"id": "first", "name": "echo", "arguments": {"text": "one"}},
                {"id": "second", "name": "echo", "arguments": {"text": "two"}},
            ]},
        ]))
        agent = make_agent()

        async def execute(tool_call_id, params, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text=params["text"])])

        agent.tools = [AgentTool(name="echo", description="echo", execute=execute)]

        def fail_first_result(event):
            if (event.type == "message_end" and event.message.role == "toolResult"
                    and event.message.tool_call_id == "first"):
                raise RuntimeError("subscriber failed")

        agent.subscribe(fail_first_result)
        try:
            with pytest.raises(RuntimeError, match="subscriber failed"):
                await agent.prompt("run tools")
            results = [message for message in agent.messages if message.role == "toolResult"]
            assert [message.tool_call_id for message in results] == ["first", "second"]
            assert [message.content[0].text for message in results] == ["one", "two"]
        finally:
            faux.reset_script_index()

    asyncio.run(scenario())
