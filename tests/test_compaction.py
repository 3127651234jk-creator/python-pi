import asyncio

import pytest

from pi.agent.agent import Agent
from pi.agent.compaction import (
    HISTORY_SUMMARIZATION_PROMPT,
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    CompactionSettings,
    ContextCompactor,
    compact_agent,
    prefix_checkpoint,
    restore_compaction,
)
from pi.ai.models import resolve_model
from pi.ai.providers import STREAM_FNS
from pi.ai.providers import SimpleStreamOptions
from pi.ai.types import (
    AssistantMessage, DoneEvent, StreamErrorEvent, TextContent, TranscriptContext, Usage, UserMessage,
)
from pi.coding_agent.session import SessionManager
from pi.coding_agent.usage import session_stats, summary_usage_extra


def response(text="summary", reason="stop", usage=None):
    return AssistantMessage(content=[TextContent(text=text)], api="faux", provider="faux",
                            model="faux", usage=usage or Usage(), stop_reason=reason, timestamp=0)


def test_compaction_preserves_transcript_and_caches_summary():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50))
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [UserMessage(content="old" * 4000, timestamp=0), UserMessage(content="latest", timestamp=1)]
        calls = []

        async def stream(model, context, options):
            calls.append(context)
            yield DoneEvent(reason="stop", message=response())

        projected = await compactor.project(messages, model, SimpleStreamOptions(), stream)
        assert projected[-1] is messages[-1]
        assert "summary" in projected[0].content
        assert len(messages[0].content) == 12000
        count = len(calls)
        assert await compactor.project(messages, model, SimpleStreamOptions(), stream) == projected
        assert len(calls) == count
    asyncio.run(check())


def test_overflow_retries_once():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=1))
        messages = [UserMessage(content="old" * 100, timestamp=0), UserMessage(content="latest", timestamp=1)]
        requests = []

        async def stream(model, context, options):
            if context.messages[0].role == "system":
                yield DoneEvent(reason="stop", message=response())
            else:
                requests.append(context)
                error = response(reason="error")
                error.error_message = "context_length_exceeded"
                yield StreamErrorEvent(reason="error", error=error)

        events = [e async for e in compactor.stream(resolve_model("faux"), TranscriptContext(messages), SimpleStreamOptions(), stream)]
        assert len(requests) == 2
        assert len(events) == 1
        assert isinstance(events[0], StreamErrorEvent)
    asyncio.run(check())


@pytest.mark.parametrize("reason,text", [("length", "partial"), ("stop", "")])
def test_failed_summary_keeps_previous_state(reason, text):
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=1))

        async def stream(model, context, options):
            yield DoneEvent(reason=reason, message=response(text, reason))

        with pytest.raises(RuntimeError):
            await compactor.project([UserMessage(content="x" * 1000, timestamp=0), UserMessage(content="latest", timestamp=1)],
                                    resolve_model("faux"), SimpleStreamOptions(), stream, force=True)
        assert compactor.prefix == []
        assert compactor.summary == ""
    asyncio.run(check())


def test_split_turn_summarizes_history_and_turn_prefix_separately():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50, summary_tokens=512))
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [
            UserMessage(content="old" * 4000, timestamp=0),
            UserMessage(content="build the parser", timestamp=1),
            AssistantMessage(content=[TextContent(text="work" * 50)], api="faux", provider="faux",
                             model="faux", usage=Usage(), stop_reason="stop", timestamp=2),
        ]
        calls = []

        async def stream(model, context, options):
            calls.append(context)
            text = "TURN_CONTEXT" if context.messages[0].content == TURN_PREFIX_SUMMARIZATION_PROMPT else "HISTORY"
            yield DoneEvent(reason="stop", message=response(text))

        projected = await compactor.project(messages, model, SimpleStreamOptions(), stream)
        # The cut lands on the assistant, so the assistant is kept and the older
        # history and the current turn's prefix are summarized by separate calls.
        assert projected[-1] is messages[-1]
        assert len(projected) == 2
        assert calls[0].messages[0].content == HISTORY_SUMMARIZATION_PROMPT
        assert calls[-1].messages[0].content == TURN_PREFIX_SUMMARIZATION_PROMPT
        assert compactor.summary == "HISTORY\n\n---\n\n**Turn Context (split turn):**\n\nTURN_CONTEXT"
    asyncio.run(check())


def test_split_turn_without_history_region_merges_prior_summary():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50, summary_tokens=512))
        compactor.summary = "prior"
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [
            UserMessage(content="req" * 8000, timestamp=0),
            AssistantMessage(content=[TextContent(text="work" * 50)], api="faux", provider="faux",
                             model="faux", usage=Usage(), stop_reason="stop", timestamp=1),
        ]
        calls = []

        async def stream(model, context, options):
            calls.append(context)
            yield DoneEvent(reason="stop", message=response("TURN_CONTEXT"))

        projected = await compactor.project(messages, model, SimpleStreamOptions(), stream)
        assert projected[-1] is messages[-1]
        # No history region: the prior summary stands in and only the turn
        # prefix is summarized.
        assert calls and all(c.messages[0].content == TURN_PREFIX_SUMMARIZATION_PROMPT for c in calls)
        assert compactor.summary == "prior\n\n---\n\n**Turn Context (split turn):**\n\nTURN_CONTEXT"
    asyncio.run(check())


def _compact_transcript():
    """Run one compaction over two messages and return (messages, compactor)."""
    async def run():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50))
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [UserMessage(content="old" * 4000, timestamp=0), UserMessage(content="latest", timestamp=1)]

        async def stream(model, context, options):
            yield DoneEvent(reason="stop", message=response())

        await compactor.project(messages, model, SimpleStreamOptions(), stream)
        return messages, compactor

    return asyncio.run(run())


def test_prefix_checkpoint_roundtrip(tmp_path):
    messages, compactor = _compact_transcript()
    checkpoint = prefix_checkpoint(compactor.prefix)
    assert checkpoint["length"] == 1

    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    for message in messages:
        manager.append(message)
    manager.append_compaction(compactor.summary, tokens_before=100, checkpoint=checkpoint)

    loaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path)).load(manager.session_file)
    fresh = ContextCompactor(CompactionSettings())
    assert restore_compaction(fresh, loaded, manager.latest_custom("compaction")) is True
    assert fresh.prefix == compactor.prefix
    assert fresh.summary == compactor.summary

    # The restored compactor must reproduce the projection without a re-summarization call.
    calls = []

    async def check():
        model = resolve_model("faux")
        model.context_window = 3000

        async def stream(model, context, options):
            calls.append(context)
            yield DoneEvent(reason="stop", message=response())

        projected = await fresh.project(loaded, model, SimpleStreamOptions(), stream)
        assert calls == []
        assert "Previous conversation summary" in projected[0].content
        assert projected[1] is loaded[-1]
        assert len(projected) == 2  # no duplicated transcript under the summary

    asyncio.run(check())


def test_restore_rejects_tampered_checkpoint(tmp_path):
    messages, compactor = _compact_transcript()
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    for message in messages:
        manager.append(message)
    tampered = {"length": prefix_checkpoint(compactor.prefix)["length"] + 1, "last": "0" * 16}
    manager.append_compaction(compactor.summary, checkpoint=tampered)

    fresh = ContextCompactor(CompactionSettings())
    fresh.summary = "stale"
    assert restore_compaction(fresh, manager.messages(), manager.latest_custom("compaction")) is False
    assert fresh.prefix == []
    assert fresh.summary == ""


def test_restore_legacy_entry_without_checkpoint(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    manager.append(UserMessage(content="hello", timestamp=0))
    manager.append_compaction("old summary", tokens_before=10)

    fresh = ContextCompactor(CompactionSettings())
    assert restore_compaction(fresh, manager.messages(), manager.latest_custom("compaction")) is False
    assert fresh.summary == "old summary"
    assert fresh.prefix == []


def test_restore_without_entry_resets_state():
    fresh = ContextCompactor(CompactionSettings())
    fresh.summary = "stale"
    fresh.prefix = [UserMessage(content="x", timestamp=0)]
    assert restore_compaction(fresh, [UserMessage(content="hello", timestamp=0)], None) is False
    assert fresh.prefix == []
    assert fresh.summary == ""


def test_summary_request_usage_is_captured_across_chunks():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50))
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [UserMessage(content="old" * 4000, timestamp=0), UserMessage(content="latest", timestamp=1)]
        calls = []

        async def stream(model, context, options):
            usage = Usage(input=10 * (len(calls) + 1), output=5 * (len(calls) + 1),
                          cost={"total": 0.01 * (len(calls) + 1)})
            calls.append(usage)
            yield DoneEvent(reason="stop", message=response(usage=usage))

        await compactor.project(messages, model, SimpleStreamOptions(), stream)
        assert calls
        # Every summarization request's usage is aggregated, however many chunks ran.
        assert compactor.last_usage.input == sum(u.input for u in calls)
        assert compactor.last_usage.output == sum(u.output for u in calls)
        assert abs(compactor.last_usage.cost["total"] - sum(u.cost["total"] for u in calls)) < 1e-9
    asyncio.run(check())


def test_split_turn_combines_history_and_turn_prefix_usage():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50, summary_tokens=512))
        model = resolve_model("faux")
        model.context_window = 3000
        messages = [
            UserMessage(content="old" * 4000, timestamp=0),
            UserMessage(content="build the parser", timestamp=1),
            AssistantMessage(content=[TextContent(text="work" * 50)], api="faux", provider="faux",
                             model="faux", usage=Usage(), stop_reason="stop", timestamp=2),
        ]
        seen = []

        async def stream(model, context, options):
            is_turn = context.messages[0].content == TURN_PREFIX_SUMMARIZATION_PROMPT
            usage = Usage(input=7, output=3, cost={"total": 0.05})
            seen.append(("turn" if is_turn else "history", usage))
            yield DoneEvent(reason="stop", message=response("TURN_CONTEXT" if is_turn else "HISTORY", usage=usage))

        await compactor.project(messages, model, SimpleStreamOptions(), stream)
        history = [u for kind, u in seen if kind == "history"]
        turn = [u for kind, u in seen if kind == "turn"]
        assert history and turn
        assert compactor.last_usage.input == sum(u.input for u in history + turn)
        assert compactor.last_usage.output == sum(u.output for u in history + turn)
        assert abs(compactor.last_usage.cost["total"] - 0.05 * len(seen)) < 1e-9
    asyncio.run(check())


def test_on_summary_receives_aggregated_usage():
    async def check():
        captured = []
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=50))
        compactor.on_summary = lambda summary, usage: captured.append((summary, usage))
        model = resolve_model("faux")
        model.context_window = 100000
        messages = [UserMessage(content="old" * 4000, timestamp=0), UserMessage(content="latest", timestamp=1)]

        async def stream(model, context, options):
            yield DoneEvent(reason="stop", message=response(usage=Usage(input=3, output=2)))

        await compactor.project(messages, model, SimpleStreamOptions(), stream, force=True)
        assert len(captured) == 1
        assert captured[0][0] == compactor.summary
        assert captured[0][1] is compactor.last_usage
        assert captured[0][1].input == 3 and captured[0][1].output == 2
    asyncio.run(check())


def test_compact_agent_returns_and_persists_summary_usage(monkeypatch, tmp_path):
    async def stream(model, context, options):
        yield DoneEvent(reason="stop", message=response(usage=Usage(input=11, output=4, cost={"total": 0.02})))

    monkeypatch.setitem(STREAM_FNS, "faux", stream)

    async def check():
        agent = Agent(resolve_model("faux"))
        agent.model.context_window = 100000
        agent.messages = [UserMessage(content="old" * 4000, timestamp=0), UserMessage(content="latest", timestamp=1)]
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        for message in agent.messages:
            session.append(message)

        result = await compact_agent(agent, session)
        assert result["usage"] is not None
        assert result["usage"].input == 11 and result["usage"].output == 4

        data = session.latest_custom("compaction")
        assert data["usage"]["input"] == 11 and data["usage"]["output"] == 4
        assert data["usage"]["cost"]["total"] == 0.02

        usages = session.compaction_usages()
        assert len(usages) == 1 and usages[0].input == 11

        stats = session_stats(agent.messages, agent.model, session.session_file,
                              extra_usage=summary_usage_extra(session))
        assert stats["tokens"]["input"] >= 11
    asyncio.run(check())


def test_compact_agent_without_compaction_reports_no_usage(monkeypatch):
    async def stream(model, context, options):
        yield DoneEvent(reason="stop", message=response(usage=Usage(input=11, output=4)))

    monkeypatch.setitem(STREAM_FNS, "faux", stream)

    async def check():
        agent = Agent(resolve_model("faux"))
        agent.model.context_window = 100000
        agent.messages = [UserMessage(content="tiny", timestamp=0)]
        result = await compact_agent(agent, session=None)
        assert result["usage"] is None
    asyncio.run(check())
