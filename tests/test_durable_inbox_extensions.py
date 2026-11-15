import asyncio
import pytest

from pi.ai.models import Model
from pi.ai.types import AssistantMessage, DoneEvent, TextContent, ToolCall, Usage
from pi.ai.utils import get_current_tools
from pi.agent.types import AgentTool, AgentToolResult
from pi.durable.extensions import AgentDoc, Extension, Registry
from pi.durable.harness import Harness
from pi.durable.harness import _stream_once
from pi.durable.inbox import InboxDoc
from pi.durable.session import Session
from pi.durable.storage import JsonlStorage


def model():
    return Model(id="faux", name="faux", api="faux", provider="faux", base_url="")


def test_queued_input_survives_reopen_and_drains(tmp_path, monkeypatch):
    async def stream(*args, **kwargs):
        return AssistantMessage(content=[TextContent(text="answer")], stop_reason="stop",
                                api="faux", provider="faux", model="faux", timestamp=0, usage=Usage())
    monkeypatch.setattr("pi.durable.harness._stream_once", stream)
    async def scenario():
        file = str(tmp_path / "state.jsonl")
        session = Session(JsonlStorage(file))
        harness = Harness(session, model())
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "first")
        queued = await harness.submit_input(cid, "second", "request")
        assert (await harness.submission(queued)).status == "queued"
        assert await harness.submit_input(cid, "second", "request") == queued
        await session.close()
        session = Session(JsonlStorage(file))
        harness = Harness(session, model())
        assert len((await harness.inbox(cid))["items"]) == 1
        await harness.run_pending_tasks()
        assert (await harness.submission(queued)).status == "done"
        assert (await harness.inbox(cid))["items"] == []
        # Turn 1: user, pi.system baseline, assistant. Turn 2 plans no diff,
        # so only user + assistant are appended.
        assert len(await harness.transcript(cid)) == 5
        await session.close()
    asyncio.run(scenario())


def test_extension_tool_hooks_and_memo(tmp_path):
    async def scenario():
        seen = []
        async def before(call, api, context):
            assert await api.memo("decision", {"x": 2}) == {"x": 2}
            assert await api.memo("decision", {"x": 3}) == {"x": 2}
            return {"arguments": {"x": 2}}
        async def execute(call_id, arguments, signal, update):
            seen.append(arguments)
            return AgentToolResult(content=[TextContent(text=str(arguments["x"]))])
        extension = Extension("demo", tools=[AgentTool(name="echo", description="echo", execute=execute)],
                              hooks={"before_tool": before})
        session = Session(JsonlStorage(str(tmp_path / "state.jsonl")))
        harness = Harness(session, model(), extensions=[extension])
        cid = await harness.create_conversation()
        assert await session.snapshot(InboxDoc, cid) == {"items": []}
        assert (await session.snapshot(AgentDoc, cid))["extensions"] == ["demo"]
        task_id = await harness.submit_tool_task(cid, "echo", {"x": 1})
        await harness.run_pending_tasks()
        assert seen == [{"x": 2}]
        assert (await harness._get_task(task_id)).state.outcome.status == "completed"
        await session.close()
    asyncio.run(scenario())


def test_abort_withdraws_users_and_preserves_writes(tmp_path):
    async def scenario():
        session = Session(JsonlStorage(str(tmp_path / "state.jsonl")))
        harness = Harness(session, model())
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "first")
        queued = await harness.submit_input(cid, "second")
        write = await harness.submit_write(cid, {"kind": "note", "data": {"value": 1}})
        await harness.abort_conversation(cid)
        assert (await harness.submission(queued)).reason == "aborted"
        assert [item["id"] for item in (await harness.inbox(cid))["items"]] == [write]
        await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("reset", [False, True])
def test_tool_round_places_steering_and_calls_after_tools(tmp_path, monkeypatch, reset):
    async def scenario():
        requests, rounds = [], []
        entered, release = asyncio.Event(), asyncio.Event()
        async def stream(model, messages, harness, **kwargs):
            requests.append(messages)
            content = [ToolCall(id="call", name="echo", arguments={})] if len(requests) == 1 else [TextContent(text="done")]
            return AssistantMessage(content=content, stop_reason="toolUse" if len(requests) == 1 else "stop",
                                    api="faux", provider="faux", model="faux", usage=Usage(), timestamp=0)
        monkeypatch.setattr("pi.durable.harness._stream_once", stream)
        async def execute(*args):
            entered.set()
            await release.wait()
            return AgentToolResult(content=[TextContent(text="tool result")])
        async def after_tools(assistant, results, api, context):
            rounds.append(results)
        extension = Extension("demo", tools=[AgentTool(name="echo", description="echo", execute=execute)],
                              hooks={"after_tools": after_tools})
        session = Session(JsonlStorage(str(tmp_path / "state.jsonl")))
        harness = Harness(session, model(), extensions=[extension])
        cid = await harness.create_conversation()
        first = await harness.submit_input(cid, "first")
        running = asyncio.create_task(harness.run_pending_tasks())
        await asyncio.wait_for(entered.wait(), 2)
        if reset:
            await harness.submit_write(cid, {"kind": "reset", "head": "self"})
        queued = await harness.submit_input(cid, "steering")
        release.set()
        await asyncio.wait_for(running, 5)
        assert len(requests) == 2
        assert len(rounds) == 1
        assert any(getattr(message, "content", None) == "steering" for message in requests[1])
        assert (await harness.submission(queued)).status == "done"
        original = await harness.submission((await harness._get_task(first)).input["submission_id"])
        assert original.status == ("unanswered" if reset else "done")
        if reset:
            assert not any(getattr(message, "content", None) == "first" for message in requests[1])
        await session.close()
    asyncio.run(scenario())


def test_generation_declares_extension_tools(monkeypatch):
    async def stream(model, context, options):
        assert [tool.name for tool in get_current_tools(context.messages)] == ["echo"]
        yield DoneEvent(reason="stop", message=AssistantMessage(content=[TextContent(text="done")],
            api="faux", provider="faux", model="faux", usage=Usage(), stop_reason="stop", timestamp=0))
    monkeypatch.setattr("pi.durable.harness.get_default_stream_fn", lambda: stream)
    result = asyncio.run(_stream_once(model(), [], None, tools=[AgentTool(name="echo", description="echo")]))
    assert result.stop_reason == "stop"
