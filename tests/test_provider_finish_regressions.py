import asyncio

import pytest

from pi.ai.models import Model
from pi.ai.providers import anthropic, openai_completions
from pi.ai.providers.base import ProviderState


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
