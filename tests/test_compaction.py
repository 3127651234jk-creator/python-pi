import asyncio

import pytest

from pi.agent.compaction import (
    CompactionSettings,
    ContextCompactor,
    prefix_checkpoint,
    restore_compaction,
)
from pi.ai.models import resolve_model
from pi.ai.providers import SimpleStreamOptions
from pi.ai.types import AssistantMessage, DoneEvent, StreamErrorEvent, TextContent, TranscriptContext, Usage, UserMessage
from pi.coding_agent.session import SessionManager


def response(text="summary", reason="stop"):
    return AssistantMessage(content=[TextContent(text=text)], api="faux", provider="faux",
                            model="faux", usage=Usage(), stop_reason=reason, timestamp=0)


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
