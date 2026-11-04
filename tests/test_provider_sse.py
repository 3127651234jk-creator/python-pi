"""Provider SSE tests: canned streams through the real event converters."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from pi.ai.models import Model
from pi.ai.types import (
    DoneEvent,
    StreamErrorEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    TranscriptContext,
    UserMessage,
)


def make_model(provider: str) -> Model:
    if provider == "anthropic":
        return Model(
            id="claude-test", name="t", api="anthropic-messages", provider="anthropic",
            base_url="https://mock", api_key="k", max_tokens=100,
        )
    return Model(
        id="gpt-test", name="t", api="openai-completions", provider="openai",
        base_url="https://mock", api_key="k", max_tokens=100,
    )


def user_context() -> TranscriptContext:
    return TranscriptContext(messages=[UserMessage(content="hi", timestamp=0)])


def stream_with(monkeypatch, provider_module, sse_body: str):
    """Patch httpx send to return a canned SSE response and collect events."""

    captured = {}

    async def fake_send(self, request, stream=False):
        captured["request"] = request
        captured["body"] = json.loads(request.content.decode("utf-8"))

        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_body.splitlines():
                    yield line

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def collect():
        fn = provider_module.stream_anthropic if provider_module.__name__.endswith("anthropic") else provider_module.stream_openai_completions
        events = []
        async for event in fn(make_model(provider_module.__name__.endswith("anthropic") and "anthropic" or "openai"), user_context()):
            events.append(event)
        return events, captured

    return collect


ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":10,"cache_read_input_tokens":5}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig1"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_1","name":"read"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"a.txt\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
"""


def test_anthropic_stream_mapping(monkeypatch):
    from pi.ai.providers import anthropic

    collect = stream_with(monkeypatch, anthropic, ANTHROPIC_SSE)
    events, captured = asyncio.run(collect())

    done = [e for e in events if isinstance(e, DoneEvent)][0]
    message = done.message
    assert done.reason == "toolUse"
    kinds = [type(b).__name__ for b in message.content]
    assert kinds == ["ThinkingContent", "TextContent", "ToolCall"]
    assert message.content[0].thinking == "hmm"
    assert message.content[0].thinking_signature == "sig1"
    assert message.content[1].text == "Hello"
    assert message.content[2].arguments == {"path": "a.txt"}
    assert message.usage.input == 10
    assert message.usage.output == 7
    assert message.usage.cache_read == 5

    # Request payload carries the folded system prompt position and tools.
    body = captured["body"]
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_request_includes_tools(monkeypatch):
    from pi.ai.providers import anthropic
    from pi.ai.types import SystemMessage, Tool

    body_holder = {}

    async def fake_send(self, request, stream=False):
        body_holder["body"] = json.loads(request.content.decode("utf-8"))

        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                return
                yield

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    context = TranscriptContext(
        messages=[
            SystemMessage(content="prompt", timestamp=0, tools_added=[Tool(name="read", description="d", parameters={"type": "object"})]),
            UserMessage(content="hi", timestamp=0),
        ]
    )

    async def run():
        events = []
        async for event in anthropic.stream_anthropic(make_model("anthropic"), context):
            events.append(event)

    asyncio.run(run())
    body = body_holder["body"]
    assert body["system"] == "prompt"
    assert body["tools"] == [{"name": "read", "description": "d", "input_schema": {"type": "object"}}]


OPENAI_SSE = """data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}

data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}

data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}

data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"x\\"}"}}]},"finish_reason":null}]}

data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":null}

data: {"id":"c1","model":"gpt-test","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}

data: [DONE]
"""


def test_openai_stream_mapping(monkeypatch):
    from pi.ai.providers import openai_completions

    collect = stream_with(monkeypatch, openai_completions, OPENAI_SSE)
    events, captured = asyncio.run(collect())

    done = [e for e in events if isinstance(e, DoneEvent)][0]
    message = done.message
    assert done.reason == "toolUse"
    kinds = [type(b).__name__ for b in message.content]
    assert kinds == ["TextContent", "ThinkingContent", "ToolCall"]
    assert message.content[0].text == "Hi"
    assert message.content[1].thinking == "think"
    assert message.content[2].arguments == {"path": "x"}
    assert message.usage.input == 11
    assert message.usage.output == 4
    assert message.usage.cache_read == 3
    assert message.usage.reasoning == 2

    body = captured["body"]
    assert body["stream"] is True
    assert body["messages"][0] == {"role": "user", "content": "hi"}


def test_anthropic_http_error_yields_error_event(monkeypatch):
    from pi.ai.providers import anthropic

    async def fake_send(self, request, stream=False):
        class FakeResponse:
            status_code = 401

            async def aiter_lines(self):
                return
                yield

            async def aread(self):
                return b'{"error":{"message":"bad key"}}'

            async def aclose(self):
                return None

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def run():
        return [e async for e in anthropic.stream_anthropic(make_model("anthropic"), user_context())]

    events = asyncio.run(run())
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, StreamErrorEvent)
    assert "401" in error.error.error_message
    assert "bad key" in error.error.error_message
