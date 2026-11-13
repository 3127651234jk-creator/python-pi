"""Provider failures and repeated terminal frames must not corrupt tool calls."""

import asyncio
import json

import httpx
import pytest

from pi.ai.models import Model
from pi.ai.providers import anthropic, openai_completions, openai_responses
from pi.ai.types import DoneEvent, StreamErrorEvent, ToolCall, TranscriptContext


def stream_events(monkeypatch, provider, payload):
    async def send(self, request, **kwargs):
        wire = "".join("data: " + (item if isinstance(item, str) else json.dumps(item)) + "\n\n" for item in payload)
        return httpx.Response(200, request=request, content=wire.encode())

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model = Model("test", "test", provider, "test", "https://example.test", api_key="test")
    fn = {"anthropic-messages": anthropic.stream_anthropic,
          "openai-completions": openai_completions.stream_openai_completions,
          "openai-responses": openai_responses.stream_openai_responses}[provider]

    async def scenario():
        return [event async for event in fn(model, TranscriptContext([]))]

    return asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["content_filter", "refusal", "unexpected"])
@pytest.mark.parametrize("api", ["openai-completions", "anthropic-messages", "openai-responses"])
def test_unsuccessful_terminal_reason_surfaces_error(monkeypatch, api, reason):
    payload = {
        "openai-completions": [{"choices": [{"delta": {}, "finish_reason": reason}]}, "[DONE]"],
        "anthropic-messages": [{"type": "message_delta", "delta": {"stop_reason": reason}}, {"type": "message_stop"}],
        "openai-responses": [{"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": reason}}}],
    }[api]
    events = stream_events(monkeypatch, api, payload)
    assert isinstance(events[-1], StreamErrorEvent)
    assert reason in events[-1].error.error_message
    assert not any(isinstance(event, DoneEvent) for event in events)


@pytest.mark.parametrize("kind", ["response.refusal.delta", "response.refusal.done"])
def test_responses_refusal_event_is_error(monkeypatch, kind):
    events = stream_events(monkeypatch, "openai-responses", [{"type": kind, "delta": "denied", "refusal": "denied"}])
    assert isinstance(events[-1], StreamErrorEvent)
    assert "denied" in events[-1].error.error_message


@pytest.mark.parametrize("kind", ["response.output_item.done", "response.completed"])
def test_responses_whole_refusal_is_error(monkeypatch, kind):
    item = {"type": "message", "content": [{"type": "refusal", "refusal": "denied"}]}
    events = stream_events(monkeypatch, "openai-responses", [
        {"type": kind, "item": item, "response": {"status": "completed", "output": [item]}},
    ])
    assert isinstance(events[-1], StreamErrorEvent)
    assert "denied" in events[-1].error.error_message


def test_anthropic_preserves_whole_tool_input(monkeypatch):
    events = stream_events(monkeypatch, "anthropic-messages", [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call", "name": "echo", "input": {"text": "hi"}}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}, {"type": "message_stop"},
    ])
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].message.content[0].arguments == {"text": "hi"}


def test_duplicate_completions_finish_keeps_arguments(monkeypatch):
    finish = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    events = stream_events(monkeypatch, "openai-completions", [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call", "function": {"name": "echo", "arguments": '{"text":"hi"}'}}]}}]},
        finish, finish, "[DONE]",
    ])
    assert isinstance(events[-1], DoneEvent)
    call = next(block for block in events[-1].message.content if isinstance(block, ToolCall))
    assert call.arguments == {"text": "hi"}
    assert sum(event.type == "toolcall_end" for event in events) == 1
