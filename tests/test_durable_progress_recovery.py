"""Durable progress end to end: partials, tool output, recovery (P08-P10)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pi.agent.types import AgentToolResult
from pi.ai.models import Model
from pi.ai.retry import RetryPolicy
from pi.ai.types import (
    AssistantMessage,
    DoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextContent,
    TextDeltaEvent,
    Usage,
)
from pi.durable.entries import ASSISTANT_ENTRY, SYSTEM_ENTRY, TOOL_RESULT_ENTRY, USER_ENTRY
from pi.durable.errors import StorageRejected
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


def partial_message() -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text="")], api="faux", provider="faux",
                            model="faux", usage=Usage(), stop_reason="pending", timestamp=0)


def chunk_stream(chunks, delay=0.01):
    async def stream(model, context, options=None):
        message = partial_message()
        yield StreamStartEvent(partial=message)
        for chunk in chunks:
            message.content[0].text += chunk
            yield TextDeltaEvent(content_index=0, delta=chunk, partial=message)
            await asyncio.sleep(delay)
        message.stop_reason = "stop"
        message.usage = Usage(input=1, output=1)
        yield DoneEvent(reason="stop", message=message)

    return stream


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached")


def install_stream(monkeypatch, stream):
    monkeypatch.setattr("pi.durable.harness.get_default_stream_fn", lambda: stream)


def test_p08_generation_partial_is_visible_then_replaced_by_the_answer(tmp_path, monkeypatch):
    install_stream(monkeypatch, chunk_stream(["Hel", "lo ", "world"], delay=0.02))

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), progress={"partial_interval_ms": 0})
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "hello")
        running = asyncio.create_task(harness.run_pending_tasks())

        async def seen_partial():
            live = await harness.live(cid)
            message = (live.get("generation") or {}).get("message")
            return bool(message) and "Hel" in json.dumps(message, ensure_ascii=False)

        await wait_until(seen_partial)
        await running

        transcript = await harness.transcript(cid)
        assert [entry.kind for entry in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        assert transcript[2].model[0].content[0].text == "Hello world"
        assert "generation" not in await harness.live(cid)
        await session.close()

    asyncio.run(scenario())


class StreamingTool:
    async def execute(self, call_id, params, signal=None, on_update=None):
        text = ""
        for index in range(30):
            text += "行😀" + "x" * 5000 + "\n"
            if on_update is not None:
                await on_update(AgentToolResult(
                    content=[TextContent(text=text)], details={"chunk": index}))
            await asyncio.sleep(0.02)
        return AgentToolResult(content=[TextContent(text=text)], details={"done": True})


def test_p08_tool_output_window_and_dropped_counts(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), progress={"output_interval_ms": 20})
        harness.register_tools({"stream": StreamingTool()})
        cid = await harness.create_conversation()
        await harness.submit_tool_task(cid, "stream", {})
        running = asyncio.create_task(harness.run_pending_tasks())

        async def saw_progress():
            live = await harness.live(cid)
            slots = live.get("tools") or []
            return bool(slots and slots[0].get("output") and slots[0].get("dropped_bytes", 0) > 0)

        await wait_until(saw_progress, timeout=15)
        live = await harness.live(cid)
        slot = live["tools"][0]
        assert slot["status"] == "running"
        assert len(slot["output"].encode("utf-8")) <= 50 * 1024
        assert slot["dropped_bytes"] > 0
        assert "😀" in slot["output"]  # UTF-8 boundaries stay intact

        await running
        transcript = await harness.transcript(cid)
        assert [entry.kind for entry in transcript] == [TOOL_RESULT_ENTRY.kind]
        assert transcript[0].model[0].content[0].text.count("行😀") == 30
        live = await harness.live(cid)
        slot = live["tools"][0]
        assert slot["status"] == "done" and slot["entry"] == transcript[0].id
        assert "output" not in slot and "dropped_bytes" not in slot
        await session.close()

    asyncio.run(scenario())


def blocking_stream(started, release, committed="committed-partial"):
    async def stream(model, context, options=None):
        message = partial_message()
        yield StreamStartEvent(partial=message)
        message.content[0].text = committed
        yield TextDeltaEvent(content_index=0, delta=committed, partial=message)
        await asyncio.sleep(0.02)
        started.set()
        await release.wait()
        message.content[0].text += " uncommitted-tail"
        yield TextDeltaEvent(content_index=0, delta=" uncommitted-tail", partial=message)
        message.stop_reason = "stop"
        yield DoneEvent(reason="stop", message=message)

    return stream


def test_p09_disk_reopen_restores_only_committed_partial_and_completes_once(tmp_path, monkeypatch):
    path = str(tmp_path / "s.jsonl")

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        install_stream(monkeypatch, blocking_stream(started, release))
        session = create_session(JsonlStorage(path))
        harness = Harness(session, make_model(), progress={"partial_interval_ms": 0})
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "hello")
        running = asyncio.create_task(harness.run_pending_tasks())

        async def saw_partial():
            live = await harness.live(cid)
            return "committed-partial" in json.dumps(live, ensure_ascii=False)

        await wait_until(saw_partial)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await session.close()

        reopened = create_session(JsonlStorage(path))
        harness2 = Harness(reopened, make_model(), progress={"partial_interval_ms": 0})
        live = await harness2.live(cid)
        assert "committed-partial" in json.dumps(live, ensure_ascii=False)
        assert "uncommitted-tail" not in json.dumps(live, ensure_ascii=False)
        assert [entry.kind for entry in await harness2.transcript(cid)] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind]

        install_stream(monkeypatch, chunk_stream(["final answer"], delay=0))
        await harness2.run_pending_tasks()
        transcript = await harness2.transcript(cid)
        assert [entry.kind for entry in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        assert transcript[2].model[0].content[0].text == "final answer"
        assert "generation" not in await harness2.live(cid)
        assert await harness2.run_pending_tasks() == 0
        await reopened.close()

    asyncio.run(scenario())


def test_p09_abort_after_reopen_converts_the_partial_once(tmp_path, monkeypatch):
    path = str(tmp_path / "s.jsonl")

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        install_stream(monkeypatch, blocking_stream(started, release))
        session = create_session(JsonlStorage(path))
        harness = Harness(session, make_model(), progress={"partial_interval_ms": 0})
        cid = await harness.create_conversation()
        task_id = await harness.submit_input(cid, "hello")
        running = asyncio.create_task(harness.run_pending_tasks())
        await wait_until(lambda: _live_has(harness, cid, "committed-partial"))
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await session.close()

        reopened = create_session(JsonlStorage(path))
        harness2 = Harness(reopened, make_model())
        await harness2.abort_task(task_id)
        await harness2.run_pending_tasks()
        transcript = await harness2.transcript(cid)
        assert [entry.kind for entry in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        assert transcript[2].model[0].content[0].text == "committed-partial"
        assert transcript[2].model[0].stop_reason == "aborted"
        assert transcript[2].data.get("partial") is True
        assert "generation" not in await harness2.live(cid)
        assert await harness2.run_pending_tasks() == 0
        assert len(await harness2.transcript(cid)) == 3
        await reopened.close()

    asyncio.run(scenario())


async def _live_has(harness, cid, text):
    return text in json.dumps(await harness.live(cid), ensure_ascii=False)


def test_p10_concurrent_conversations_keep_separate_live_state(tmp_path, monkeypatch):
    install_stream(monkeypatch, chunk_stream(["a", "b", "c", "d"], delay=0.02))

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), progress={"partial_interval_ms": 0})
        first = await harness.create_conversation()
        second = await harness.create_conversation()
        await harness.submit_input(first, "one")
        await harness.submit_input(second, "two")
        running = asyncio.create_task(harness.run_pending_tasks(concurrency=2))

        async def both_have_partials():
            one = await harness.live(first)
            two = await harness.live(second)
            return bool(one.get("generation")) and bool(two.get("generation"))

        await wait_until(both_have_partials)
        await running
        for cid in (first, second):
            transcript = await harness.transcript(cid)
            assert [entry.kind for entry in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
            assert transcript[2].model[0].content[0].text == "abcd"
            assert "generation" not in await harness.live(cid)
        await session.close()

    asyncio.run(scenario())


def test_p10_retry_discards_the_failed_partial(tmp_path, monkeypatch):
    attempts = {"count": 0}

    async def stream(model, context, options=None):
        attempts["count"] += 1
        attempt = attempts["count"]
        message = partial_message()
        yield StreamStartEvent(partial=message)
        message.content[0].text = f"partial-{attempt}"
        yield TextDeltaEvent(content_index=0, delta=f"partial-{attempt}", partial=message)
        await asyncio.sleep(0.02)
        if attempt == 1:
            message.stop_reason = "error"
            message.error_message = "503 overloaded"
            yield StreamErrorEvent(reason="error", error=message)
            return
        message.stop_reason = "stop"
        yield DoneEvent(reason="stop", message=message)

    install_stream(monkeypatch, stream)

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), retry=RetryPolicy(max_retries=1, base_delay_ms=0),
                          progress={"partial_interval_ms": 0})
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "hello")
        await harness.run_pending_tasks()
        assert attempts["count"] == 2
        transcript = await harness.transcript(cid)
        assert [entry.kind for entry in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        assert transcript[2].model[0].content[0].text == "partial-2"
        assert "generation" not in await harness.live(cid)
        await session.close()

    asyncio.run(scenario())


class RejectingStorage(JsonlStorage):
    """Rejects the tool result entry, so the final transaction fails."""

    async def commit(self, writes):
        for write in writes:
            if write.get("type") == "entry" and getattr(write["value"], "kind", None) == TOOL_RESULT_ENTRY.kind:
                raise StorageRejected("final transaction rejected")
        return await super().commit(writes)


def test_p07_final_transaction_failure_is_observable_and_leaves_no_entry(tmp_path):
    async def scenario():
        session = create_session(RejectingStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        harness.register_tools({"echo": StreamingTool()})
        cid = await harness.create_conversation()
        await harness.submit_tool_task(cid, "echo", {})
        await harness.run_pending_tasks()

        tasks = await harness.list_tasks(cid)
        assert tasks[0].state.status == "terminal"
        assert tasks[0].state.outcome.status == "faulted"
        assert "final transaction rejected" in tasks[0].state.outcome.error["message"]
        assert await harness.transcript(cid) == []
        live = await harness.live(cid)
        assert live["tools"][0]["status"] == "done"
        assert "output" not in live["tools"][0]
        await session.close()

    asyncio.run(scenario())


def test_p10_policies_are_per_harness(tmp_path):
    async def scenario():
        first_session = create_session(JsonlStorage(str(tmp_path / "a.jsonl")))
        second_session = create_session(JsonlStorage(str(tmp_path / "b.jsonl")))
        first = Harness(first_session, make_model(), progress={"partial_interval_ms": 0,
                                                               "output_interval_ms": 250})
        second = Harness(second_session, make_model(), progress={"partial_interval_ms": 500})
        assert first.progress.partial_interval_ms == 0
        assert first.progress.output_interval_ms == 250
        assert second.progress.partial_interval_ms == 500
        assert second.progress.output_interval_ms == 100
        await first_session.close()
        await second_session.close()

    asyncio.run(scenario())
