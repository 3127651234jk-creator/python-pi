import asyncio

import pytest

from pi.ai.models import Model
from pi.ai.providers import anthropic, openai_completions
from pi.ai.providers.base import ProviderState
from pi.ai.types import ToolCall, ToolCallDeltaEvent, ToolCallStartEvent


def state():
    return ProviderState(Model(id="test", name="test", api="openai-completions", provider="openai",
                               base_url="https://mock", max_tokens=100), 0)


@pytest.mark.parametrize("provider,reason", [(openai_completions, "content_filter"),
    (openai_completions, "unknown"), (anthropic, "refusal"), (anthropic, "unknown")])
def test_invalid_finish_is_error(provider, reason):
    async def check():
        current = state()
        if provider is anthropic:
            current.set_raw_stop_reason(reason)
            data = {"type": "message_stop"}
        else:
            data = {"choices": [{"finish_reason": reason}]}
        with pytest.raises(RuntimeError, match=reason):
            _ = [event async for event in provider._convert_events(None, data, current)]
    asyncio.run(check())


def test_anthropic_initial_tool_input_survives_stop():
    async def check():
        current = state()
        for data in [{"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "call", "name": "read", "input": {"path": "file"}}},
            {"type": "content_block_stop", "index": 0}]:
            _ = [event async for event in anthropic._convert_events(None, data, current)]
        assert current.partial.content[0].arguments == {"path": "file"}
    asyncio.run(check())


@pytest.mark.parametrize("reason,expected", [("end_turn", "stop"), ("tool_use", "toolUse"),
                                               ("max_tokens", "length")])
def test_anthropic_message_delta_marks_the_partial_complete_without_message_stop(reason, expected):
    async def check():
        current = state()
        events = [event async for event in anthropic._convert_events(
            None, {"type": "message_delta", "delta": {"stop_reason": reason}}, current)]
        assert events == []
        assert current.partial.raw_stop_reason == reason
        assert current.partial.stop_reason == expected

    asyncio.run(check())


def test_duplicate_finish_preserves_tool_arguments():
    async def check():
        current = state()
        chunks = [{"delta": {"tool_calls": [{"index": 0, "id": "call", "function": {
            "name": "read", "arguments": '{"path":"file"}'}}]}},
            {"finish_reason": "tool_calls"}, {"finish_reason": "tool_calls"}]
        for chunk in chunks:
            _ = [event async for event in openai_completions._convert_events(None, {"choices": [chunk]}, current)]
        assert current.partial.content[0].arguments == {"path": "file"}
    asyncio.run(check())

async def _drive_completions(current, chunks):
    events = []
    for chunk in chunks:
        async for event in openai_completions._convert_events(None, {"choices": [chunk]}, current):
            events.append(event)
    return events


def test_completions_sparse_tool_call_index_maps_by_stream_slot():
    async def check():
        current = state()
        await _drive_completions(current, [
            {"delta": {"tool_calls": [{"index": 3, "id": "call_b", "function": {
                "name": "read", "arguments": '{"path":'}}]}},
            {"delta": {"tool_calls": [{"index": 3, "function": {"arguments": '"b"}'}}]}},
            {"finish_reason": "tool_calls"},
        ])
        calls = [block for block in current.partial.content if isinstance(block, ToolCall)]
        assert len(calls) == 1
        assert calls[0].id == "call_b"
        assert calls[0].arguments == {"path": "b"}
    asyncio.run(check())


def test_completions_out_of_order_tool_call_indices_keep_arguments_apart():
    async def check():
        current = state()
        events = await _drive_completions(current, [
            {"delta": {"tool_calls": [{"index": 1, "id": "call_1", "function": {
                "name": "read", "arguments": '{"path":"one"}'}}]}},
            {"delta": {"tool_calls": [{"index": 0, "id": "call_0", "function": {
                "name": "read", "arguments": '{"path":"'}}]}},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'zero"}'}}]}},
            {"finish_reason": "tool_calls"},
        ])
        calls = [block for block in current.partial.content if isinstance(block, ToolCall)]
        assert [(call.id, call.arguments) for call in calls] == [
            ("call_1", {"path": "one"}),
            ("call_0", {"path": "zero"}),
        ]
        # Event content indexes must point at the block's real position, not
        # the provider's sparse tool-call slot.
        starts = {event.partial.content[event.content_index].id: event.content_index for event in events
                  if isinstance(event, ToolCallStartEvent)}
        assert starts == {"call_1": 0, "call_0": 1}
        deltas = [event.content_index for event in events if isinstance(event, ToolCallDeltaEvent)]
        assert deltas == [0, 1, 1]
    asyncio.run(check())


@pytest.mark.parametrize("reason,expected", [("content_filter", "error"), ("unknown", "error"),
                                             (None, "error"), ("max_output_tokens", "length")])
def test_responses_incomplete_reason(monkeypatch, reason, expected):
    from pi.ai.providers import openai_responses
    from pi.ai.types import TranscriptContext

    async def drive(model, request, path, headers, convert, options, timestamp):
        current = state()
        data = {"type": "response.incomplete", "response": {
            "status": "incomplete", "incomplete_details": {"reason": reason}}}
        try:
            async for event in convert(None, data, current):
                yield event
        except RuntimeError as exc:
            assert expected == "error"
            assert "Response incomplete" in str(exc)
            return
        assert expected == "length"

    monkeypatch.setattr(openai_responses, "run_provider_stream", drive)

    async def check():
        events = [event async for event in openai_responses.stream_openai_responses(
            state().model, TranscriptContext(messages=[]))]
        if expected == "length":
            assert events[-1].reason == "length"
        else:
            assert not events
    asyncio.run(check())
