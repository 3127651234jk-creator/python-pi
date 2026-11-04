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
    # prompt_tokens 11 includes 3 cached tokens, reported separately.
    assert message.usage.input == 8
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


def _capture_body_and_headers(monkeypatch):
    holder = {}

    async def fake_send(self, request, stream=False):
        holder["body"] = json.loads(request.content.decode("utf-8"))
        holder["headers"] = request.headers

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
    return holder


def _tools_context():
    from pi.ai.types import SystemMessage, Tool

    return TranscriptContext(
        messages=[
            SystemMessage(content="prompt", timestamp=0,
                          tools_added=[Tool(name="read", description="d", parameters={"type": "object"})]),
            UserMessage(content="hi", timestamp=0),
        ]
    )


@pytest.mark.parametrize("retention,ttl", [("5m", None), ("1h", "1h")])
def test_anthropic_cache_control_breakpoints(monkeypatch, retention, ttl):
    from pi.ai.providers import anthropic
    from pi.ai.providers.base import SimpleStreamOptions

    holder = _capture_body_and_headers(monkeypatch)
    options = SimpleStreamOptions(cache_retention=retention)

    async def run():
        async for _ in anthropic.stream_anthropic(make_model("anthropic"), _tools_context(), options):
            pass

    asyncio.run(run())
    body = holder["body"]
    marker = {"type": "ephemeral"} if ttl is None else {"type": "ephemeral", "ttl": "1h"}
    assert body["system"] == [{"type": "text", "text": "prompt", "cache_control": marker}]
    assert body["tools"][-1]["cache_control"] == marker
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "hi", "cache_control": marker}]
    if ttl == "1h":
        assert holder["headers"].get("anthropic-beta") == "extended-cache-ttl-2025-04-11"
    else:
        assert holder["headers"].get("anthropic-beta") is None


ANTHROPIC_CACHE_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":10,"cache_read_input_tokens":2,"cache_creation":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":4}}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}
"""


def test_anthropic_cache_creation_usage_split(monkeypatch):
    from pi.ai.providers import anthropic

    collect = stream_with(monkeypatch, anthropic, ANTHROPIC_CACHE_SSE)
    events, _ = asyncio.run(collect())

    done = [e for e in events if isinstance(e, DoneEvent)][0]
    assert done.message.usage.cache_read == 2
    assert done.message.usage.cache_write == 3
    assert done.message.usage.cache_write_1h == 4


def test_openai_deepseek_cache_fallback_and_cost(monkeypatch):
    from pi.ai.models import ModelCost
    from pi.ai.providers import openai_completions

    sse = """data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":null}

data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"prompt_cache_hit_tokens":3}}

data: [DONE]
"""
    async def fake_send(self, request, stream=False):
        class FakeResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse.splitlines():
                    yield line

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def run():
        model = make_model("openai")
        model.costs = ModelCost(input=1.0)
        events = []
        async for event in openai_completions.stream_openai_completions(model, user_context()):
            events.append(event)
        return events

    events = asyncio.run(run())
    done = [e for e in events if isinstance(e, DoneEvent)][0]
    assert done.message.usage.cache_read == 3
    assert done.message.usage.input == 8  # prompt_tokens 11 minus 3 cached
    assert done.message.usage.cost["input"] == pytest.approx(8 * 1.0 / 1_000_000)
    assert done.message.usage.cost["total"] == pytest.approx(8 * 1.0 / 1_000_000)
