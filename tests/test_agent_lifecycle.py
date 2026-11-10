"""Regression tests for event-stream task ownership and failure cleanup."""

import asyncio

import pytest

from pi.agent.agent import Agent
from pi.agent.types import AgentStartEvent
from pi.ai.models import Model


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
