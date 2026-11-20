"""Agent-level wiring for the 0.87.x capabilities and per-queue modes (F-AG-12/F-AG-13)."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.agent import Agent
from pi.agent.types import AgentTool, AgentToolResult, AgentTurnDecision
from pi.ai.models import Model
from pi.ai.providers import STREAM_FNS, faux
from pi.ai.types import TextContent


@pytest.fixture(autouse=True)
def faux_script_env():
    yield
    os.environ.pop("FAUX_SCRIPT", None)
    faux.reset_script_index()


def use_script(script):
    os.environ["FAUX_SCRIPT"] = json.dumps(script)
    faux.reset_script_index()


def make_agent(*model_args, **kwargs) -> Agent:
    if not model_args:
        model_args = (Model(id="faux", name="Faux", api="faux", provider="faux", base_url=""),)
    return Agent(*model_args, **kwargs)


def assistant_texts(messages):
    return [block.text for m in messages
            if getattr(m, "role", None) == "assistant"
            for block in m.content if isinstance(block, TextContent)]


def test_finish_turn_end_stops_after_first_response(monkeypatch):
    """finish_turn {action: end} ends the run without consuming the next turn."""
    use_script([{"text": "one"}, {"text": "two"}])
    decisions = []

    async def finish_turn(context, signal=None):
        decisions.append(context.message.content[0].text)
        return AgentTurnDecision(action="end")

    agent = make_agent(finish_turn=finish_turn)
    messages = asyncio.run(agent.prompt("hello"))
    assert decisions == ["one"]
    assert assistant_texts(messages) == ["one"]
    assert not agent.is_streaming


def test_finish_turn_continue_runs_another_turn(monkeypatch):
    use_script([{"text": "one"}, {"text": "two"}])

    async def finish_turn(context, signal=None):
        return AgentTurnDecision(action="continue" if context.message.content[0].text == "one" else "end")

    agent = make_agent(finish_turn=finish_turn)
    messages = asyncio.run(agent.prompt("hello"))
    assert assistant_texts(messages) == ["one", "two"]


def test_finish_turn_without_decision_keeps_loop_semantics(monkeypatch):
    """No finish_turn (or a None decision) keeps steering/continuation behavior."""
    use_script([{"text": "one"}, {"text": "two"}])

    agent = make_agent()
    messages = asyncio.run(agent.prompt("hello"))
    assert assistant_texts(messages) == ["one"]


def test_invalid_queue_mode_rejected():
    with pytest.raises(ValueError):
        make_agent(steering_queue_mode="soon")
    with pytest.raises(ValueError):
        make_agent(follow_up_queue_mode="soon")


def test_queue_mode_alias_sets_both_queues():
    agent = make_agent()
    assert agent.steering_queue_mode == "one-at-a-time"
    assert agent.follow_up_queue_mode == "one-at-a-time"
    agent.queue_mode = "all"
    assert agent.steering_queue_mode == "all"
    assert agent.follow_up_queue_mode == "all"
    agent = make_agent(queue_mode="all")
    assert agent.steering_queue_mode == "all" and agent.follow_up_queue_mode == "all"


def test_peek_queued_messages_previews_without_consuming():
    agent = make_agent()
    agent.steer("first")
    agent.steer("second")
    agent.follow_up("later")
    assert [m.content for m in agent.peek_queued_messages()] == ["first"]
    agent.steering_queue_mode = "all"
    assert [m.content for m in agent.peek_queued_messages()] == ["first", "second"]
    agent._steering_queue.clear()
    assert [m.content for m in agent.peek_queued_messages()] == ["later"]
    agent._follow_up_queue.clear()
    assert agent.peek_queued_messages() == []
    # Nothing was consumed by any peek.
    assert len(agent._steering_queue) == 0 and len(agent._follow_up_queue) == 0


def test_one_at_a_time_delivers_one_steered_message_per_turn(monkeypatch):
    """Two mid-run steers are injected across two turns, one per request."""
    use_script([{"toolCalls": [{"id": "c1", "name": "echo", "arguments": {}}]},
                {"toolCalls": [{"id": "c2", "name": "echo", "arguments": {}}]},
                {"text": "done"}])
    requests = []

    async def stream(model, context, options):
        requests.append([getattr(m, "content", None) for m in context.messages
                         if getattr(m, "role", None) == "user"])
        async for event in faux.stream_faux(model, context, options):
            yield event

    monkeypatch.setitem(STREAM_FNS, "faux", stream)

    async def scenario():
        entered = asyncio.Event()

        async def execute(tool_call_id, params, signal=None, on_update=None):
            if not entered.is_set():
                entered.set()
                agent.steer("steer one")
                agent.steer("steer two")
            return AgentToolResult(content=[TextContent(text="ok")])

        agent.tools = [AgentTool(name="echo", description="echo", execute=execute)]
        messages = await agent.prompt("go")
        assert assistant_texts(messages) == ["done"]
        assert requests[0] == ["go"]
        assert requests[1] == ["go", "steer one"]
        assert requests[2] == ["go", "steer one", "steer two"]

    agent = make_agent()
    asyncio.run(scenario())


def test_all_mode_delivers_every_steered_message_in_one_turn(monkeypatch):
    use_script([{"toolCalls": [{"id": "c1", "name": "echo", "arguments": {}}]},
                {"text": "done"}])
    requests = []

    async def stream(model, context, options):
        requests.append([getattr(m, "content", None) for m in context.messages
                         if getattr(m, "role", None) == "user"])
        async for event in faux.stream_faux(model, context, options):
            yield event

    monkeypatch.setitem(STREAM_FNS, "faux", stream)

    async def scenario():
        entered = asyncio.Event()

        async def execute(tool_call_id, params, signal=None, on_update=None):
            if not entered.is_set():
                entered.set()
                agent.steer("steer one")
                agent.steer("steer two")
            return AgentToolResult(content=[TextContent(text="ok")])

        agent.tools = [AgentTool(name="echo", description="echo", execute=execute)]
        messages = await agent.prompt("go")
        assert assistant_texts(messages) == ["done"]
        assert requests[1] == ["go", "steer one", "steer two"]

    agent = make_agent(queue_mode="all")
    asyncio.run(scenario())


def test_on_provider_stream_event_is_forwarded(monkeypatch):
    """The Agent hook reaches the stream options of every provider request."""
    use_script([{"text": "hello"}])

    async def stream(model, context, options):
        assert options.on_provider_stream_event is not None
        async for event in faux.stream_faux(model, context, options):
            yield event

    monkeypatch.setitem(STREAM_FNS, "faux", stream)
    agent = make_agent(on_provider_stream_event=lambda event: None)
    asyncio.run(agent.prompt("hello"))


def test_on_provider_stream_event_observes_http_driver_payloads():
    """The hook sits before conversion in the provider stream (raw event data)."""

    async def scenario():
        from pi.ai.types import DoneEvent
        from pi.ai.providers.base import make_partial, _call_hook

        model = Model(id="wired", name="w", api="wired-test", provider="test", base_url="")
        final = make_partial(model, 0)
        final.stop_reason = "stop"

        async def scripted(model, context, options=None):
            if options is not None and options.on_provider_stream_event is not None:
                await _call_hook(options.on_provider_stream_event,
                                 {"type": "response.completed", "raw": True})
            yield DoneEvent(reason="stop", message=final)

        STREAM_FNS["wired-test"] = scripted
        try:
            seen = []
            agent = make_agent(model, on_provider_stream_event=seen.append)
            await agent.prompt("hello")
            assert seen == [{"type": "response.completed", "raw": True}]
        finally:
            STREAM_FNS.pop("wired-test", None)

    asyncio.run(scenario())
