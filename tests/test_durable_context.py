"""Conversation context reads: at-cutoff views, the incremental cache, retention (F-AG-04/06/09)."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pi.durable.harness as harness_module
from pi.ai.models import Model
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from pi.durable.context import ContextView, order_tool_results
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage


def make_model(context_window=None) -> Model:
    model = Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")
    if context_window is not None:
        model.context_window = context_window
    return model


def answer_message(text: str = "answer") -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], api="faux", provider="faux",
                            model="faux", usage=Usage(), stop_reason="stop", timestamp=0)


async def run_turn(harness, cid, text="hello"):
    async def stream(model, messages, harness_arg, **kwargs):
        return answer_message()

    original = harness_module._stream_once
    harness_module._stream_once = stream
    try:
        await harness.submit_input(cid, text)
        await harness.run_pending_tasks()
    finally:
        harness_module._stream_once = original


def test_context_view_derives_messages_and_cuts_at(tmp_path):
    """context() returns the derived model context; `at` cuts it at a visible entry."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "hello")
        entries = await harness.transcript(cid)

        view = await harness.context(cid)
        assert isinstance(view, ContextView)
        assert view.head is None
        assert [entry.id for entry in view.entries] == [entry.id for entry in entries]
        roles = [getattr(m, "role", "") for m in view.messages]
        # The baseline system message leads; user input and the answer follow.
        assert roles == ["system", "user", "assistant"]
        assert view.messages[-1].content[0].text == "answer"

        # Cut off at the user entry: the answer and its system baseline drop.
        user_entry = next(entry for entry in entries if entry.kind == "pi.user")
        cut = await harness.context(cid, at=user_entry.id)
        assert [getattr(m, "role", "") for m in cut.messages] == ["user"]
        assert cut.entries == [user_entry]
        await session.close()

    asyncio.run(scenario())


def test_context_at_invisible_entry_is_rejected(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "hello")
        try:
            await harness.context(cid, at=999_999)
            raise AssertionError("invisible entry accepted")
        except Exception as exc:  # noqa: BLE001
            assert "not visible" in str(exc)
        await session.close()

    asyncio.run(scenario())


def test_context_cache_extends_incrementally(tmp_path):
    """A second read with the same head marker extends the cached range instead
    of rescanning it; the derived view matches a full derivation."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "first")

        first = await harness.context(cid)
        cached = harness._context_cache[cid]
        full = await harness.context(cid)
        assert harness._context_cache[cid] is cached  # same tail: range reused

        await run_turn(harness, cid, "second")
        second = await harness.context(cid)
        extended = harness._context_cache[cid]
        assert extended is not cached  # the range grew
        assert extended.entries[: len(cached.entries)] == cached.entries

        # The incremental view equals a fresh derivation of the whole range.
        fresh_harness = Harness(session, make_model())
        fresh = await fresh_harness.context(cid)
        assert [getattr(m, "role", "") for m in second.messages] == [
            getattr(m, "role", "") for m in fresh.messages]
        assert [text_of(m) for m in second.messages if getattr(m, "role", "") == "user"] == [
            "first", "second"]
        assert first.messages[-1].content[0].text == "answer"
        await session.close()

    asyncio.run(scenario())


def text_of(message):
    content = message.content
    return content if isinstance(content, str) else content[0].text


def test_task_runtime_context_shares_the_harness_read(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "hello")
        entries = await harness.transcript(cid)
        user_entry = next(entry for entry in entries if entry.kind == "pi.user")

        runtime = TaskRuntime(harness, 0, asyncio.Event())
        view = await runtime.context(cid, at=user_entry.id)
        assert [getattr(m, "role", "") for m in view.messages] == ["user"]
        await session.close()

    asyncio.run(scenario())


def test_retention_zero_drops_idle_cache_and_busy_conversations_keep_it(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), context_retention_ms=0)
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "hello")
        # The turn's drain settled idleness: retention 0 dropped the range.
        assert cid not in harness._context_cache

        # A read repopulates it; the next idle drain drops it again.
        await harness.context(cid)
        assert cid in harness._context_cache
        await harness.run_pending_tasks()
        assert cid not in harness._context_cache
        await session.close()

    asyncio.run(scenario())


def test_retention_default_keeps_the_cache(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        await run_turn(harness, cid, "hello")
        assert cid in harness._context_cache
        await harness.run_pending_tasks()
        assert cid in harness._context_cache
        await session.close()

    asyncio.run(scenario())


def test_invalid_retention_value_is_rejected():
    import pytest

    async def scenario():
        session = create_session(JsonlStorage(str(os.devnull)))
        with pytest.raises(ValueError):
            Harness(session, make_model(), context_retention_ms=-1)
        with pytest.raises(ValueError):
            Harness(session, make_model(), context_retention_ms="soon")
        await session.close()

    asyncio.run(scenario())


def test_order_tool_results_synthesizes_missing_results():
    assistant = AssistantMessage(
        content=[ToolCall(id="call-1", name="bash", arguments={})],
        stop_reason="toolUse", api="faux", provider="faux", model="faux",
        usage=Usage(), timestamp=7)
    result = ToolResultMessage(tool_call_id="call-1", tool_name="bash", is_error=False,
                               content=[TextContent(text="out")], timestamp=8)
    later_user = UserMessage(content="next", timestamp=9)

    ordered = order_tool_results([assistant, result, later_user])
    assert [getattr(m, "role", "") for m in ordered] == ["assistant", "toolResult", "user"]
    assert ordered[1] is result

    # A missing result is synthesized with the fixed text, in call order.
    ordered = order_tool_results([assistant, later_user])
    assert [getattr(m, "role", "") for m in ordered] == ["assistant", "toolResult", "user"]
    assert ordered[1].content[0].text == "Tool result unavailable: history ends before this call completed."
    assert ordered[1].is_error is True
    assert ordered[1].timestamp == 7

    # Unmatched results (a different call id) are dropped.
    stray = ToolResultMessage(tool_call_id="other", tool_name="x", is_error=False,
                              content=[TextContent(text="stray")], timestamp=10)
    ordered = order_tool_results([assistant, stray])
    assert [getattr(m, "role", "") for m in ordered] == ["assistant", "toolResult"]
    assert ordered[1].tool_call_id == "call-1"
