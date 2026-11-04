import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pi.ai.models import Model, resolve_model
from pi.ai.types import (TextContent, ToolCall, Tool, SystemMessage, UserMessage,
    ToolResultMessage, TranscriptContext, DoneEvent, StreamErrorEvent)
from pi.ai.json_io import to_json, message_from_json
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.providers import openai_responses, google, amazon_bedrock


def context():
    return TranscriptContext([SystemMessage(content="system", timestamp=0, tools_added=[Tool("echo", "Echo", {"type": "object"})]), UserMessage(content="hello", timestamp=0)])


@pytest.mark.parametrize("module,api,payload", [
    (openai_responses, "openai-responses", [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "call_id": "call", "name": "echo", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"text":"hello"}'},
        {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": '{"text":"hello"}'},
        {"type": "response.completed", "response": {"id": "response", "usage": {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 2}}}},
    ]),
    (google, "google-generative-ai", [
        {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"id": "call", "name": "echo", "args": {"text": "hello"}}, "thoughtSignature": "signed"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "cachedContentTokenCount": 2}},
    ]),
])
def test_sse_native_mapping(monkeypatch, module, api, payload):
    captured = {}
    async def send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, request=request, content="".join("data: " + json.dumps(item) + "\n\n" for item in payload).encode())
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model = Model(id="test", name="test", api=api, provider="test", base_url="https://example.test", api_key="key")
    async def check():
        stream = module.stream_google if module is google else module.stream_openai_responses
        return [event async for event in stream(model, context())]
    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert done.reason == done.message.stop_reason == "toolUse"
    assert done.message.content[0].text == "Hello"
    call = next(block for block in done.message.content if isinstance(block, ToolCall))
    assert call.arguments == {"text": "hello"}
    assert done.message.usage.input == 8 and done.message.usage.cache_read == 2
    if module is google:
        restored = message_from_json(to_json(done.message))
        assert restored.content[-1].thought_signature == "signed"
        body = google.build_request(model, TranscriptContext([*context().messages, restored]), SimpleStreamOptions())
        assert body["contents"][-1]["parts"][-1]["thoughtSignature"] == "signed"
    else:
        assert len([event for event in events if event.type == "toolcall_delta"]) == 1
        assert captured["body"]["tools"][0]["name"] == "echo"


def test_bedrock_stream_request_usage_and_cleanup(monkeypatch):
    class Stream:
        closed = False
        def __iter__(self):
            return iter([
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
                {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "call", "name": "echo"}}}},
                {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"text":"hi"}'}}}},
                {"messageStop": {"stopReason": "tool_use"}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ])
        def close(self): self.closed = True
    class Client:
        closed = False
        def converse_stream(self, **kwargs):
            self.request = kwargs
            return {"stream": stream}
        def close(self): self.closed = True
    stream, client = Stream(), Client()
    monkeypatch.setattr(amazon_bedrock, "boto3", SimpleNamespace(client=lambda *args, **kwargs: client))
    model = resolve_model("amazon-bedrock")
    async def check():
        return [event async for event in amazon_bedrock.stream_bedrock(model, context())]
    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert done.message.content[-1].arguments == {"text": "hi"}
    assert done.message.usage.output == 5 and done.reason == "toolUse"
    assert client.request["toolConfig"]["tools"][0]["toolSpec"]["name"] == "echo"
    assert stream.closed and client.closed


def test_bedrock_cache_points_in_request():
    model = Model(id="anthropic.claude-sonnet-4-5", name="t", api="bedrock-converse-stream",
                  provider="amazon-bedrock", base_url="https://bedrock.us-east-1.amazonaws.com")
    body = amazon_bedrock.build_request(model, context(), SimpleStreamOptions(cache_retention="5m"))
    assert body["system"] == [{"text": "system"}, {"cachePoint": {"type": "default"}}]
    assert body["messages"][-1]["content"][-1] == {"cachePoint": {"type": "default"}}
    long = amazon_bedrock.build_request(model, context(), SimpleStreamOptions(cache_retention="1h"))
    assert long["system"][-1] == {"cachePoint": {"type": "default", "ttl": "1h"}}


def test_bedrock_cache_points_skipped_for_other_models():
    model = Model(id="amazon.nova-pro-v1", name="t", api="bedrock-converse-stream",
                  provider="amazon-bedrock", base_url="https://bedrock.us-east-1.amazonaws.com")
    body = amazon_bedrock.build_request(model, context(), SimpleStreamOptions(cache_retention="5m"))
    assert "cachePoint" not in json.dumps(body)


def test_openai_responses_params_passthrough():
    model = Model(id="test", name="test", api="openai-responses", provider="test",
                  base_url="https://example.test", api_key="key", params={"prompt_cache_key": "session-1"})
    body = openai_responses.build_request(model, context(), SimpleStreamOptions())
    assert body["prompt_cache_key"] == "session-1"


def test_bedrock_cache_details_usage(monkeypatch):
    class Stream:
        closed = False
        def __iter__(self):
            return iter([
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 2,
                                        "cacheWriteInputTokens": 6,
                                        "cacheDetails": [{"ttl": "1h", "inputTokens": 4}]}}},
            ])
        def close(self): self.closed = True
    class Client:
        closed = False
        def converse_stream(self, **kwargs):
            self.request = kwargs
            return {"stream": Stream()}
        def close(self): self.closed = True
    monkeypatch.setattr(amazon_bedrock, "boto3", SimpleNamespace(client=lambda *args, **kwargs: Client()))
    async def check():
        return [event async for event in amazon_bedrock.stream_bedrock(resolve_model("amazon-bedrock"), context())]
    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert done.message.usage.cache_read == 2
    # The flat cache write total includes the 1h tokens; they are split out.
    assert done.message.usage.cache_write == 2
    assert done.message.usage.cache_write_1h == 4


def test_bedrock_optional_dependency_error(monkeypatch):
    monkeypatch.setattr(amazon_bedrock, "boto3", None)
    async def check():
        return [event async for event in amazon_bedrock.stream_bedrock(resolve_model("amazon-bedrock"), context())]
    events = asyncio.run(check())
    assert isinstance(events[-1], StreamErrorEvent)
    assert "bedrock" in events[-1].error.error_message


@pytest.mark.parametrize("module,api", [(google, "google-generative-ai"), (openai_responses, "openai-responses")])
def test_native_http_error_and_premature_end(monkeypatch, module, api):
    async def send(self, request, **kwargs):
        return httpx.Response(403, request=request, content=b"denied")
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async def check():
        fn = module.stream_google if module is google else module.stream_openai_responses
        return [event async for event in fn(Model("test", "test", api, "test", "https://example.test"), context())]
    assert isinstance(asyncio.run(check())[-1], StreamErrorEvent)
