import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pi.ai.models import Model, resolve_model
from pi.ai.types import (TextContent, ImageContent, AudioContent, VideoContent, ToolCall, Tool,
    SystemMessage, UserMessage, AssistantMessage, ToolResultMessage, TranscriptContext,
    DoneEvent, StreamErrorEvent, Usage)
from pi.ai.json_io import to_json, message_from_json
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.providers import openai_responses, google, amazon_bedrock, openai_completions, anthropic
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry


def context():
    return TranscriptContext([SystemMessage(content="system", timestamp=0, tools_added=[Tool("echo", "Echo", {"type": "object"})]), UserMessage(content="hello", timestamp=0)])


def assistant(content, api="openai-responses"):
    return AssistantMessage(content=content, api=api, provider="test", model="test",
        usage=Usage(), stop_reason="stop", timestamp=0)


@pytest.mark.parametrize("entra", [False, True])
def test_azure_endpoint_auth_and_transport_params(tmp_path, monkeypatch, entra):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("PI_PY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_ACCESS_TOKEN", raising=False)
    if entra:
        monkeypatch.setenv("AZURE_ACCESS_TOKEN", "entra-token")
    registry = ModelRegistry(Configuration(tmp_path, tmp_path / "agent"))
    model = registry.resolve("azure-openai", "deployment/name")
    model.params = {"api_version": "2024-10-21"}
    captured = {}
    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=b'data: [DONE]\n\n')
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async def check():
        key = await model.get_api_key(model.provider)
        return [event async for event in openai_completions.stream_openai_completions(
            model, context(), SimpleStreamOptions(api_key=key))]
    assert isinstance(asyncio.run(check())[-1], DoneEvent)
    request = captured["request"]
    assert request.url.raw_path == b"/openai/deployments/deployment%2Fname/chat/completions?api-version=2024-10-21"
    assert "api_version" not in json.loads(request.content)
    if entra:
        assert request.headers["authorization"] == "Bearer entra-token"
        assert "api-key" not in request.headers
    else:
        assert request.headers["api-key"] == "azure-key"


@pytest.mark.parametrize("module,api,params,field", [
    (openai_completions, "openai-completions", {"reasoning_effort": "low"}, "reasoning_effort"),
    (openai_responses, "openai-responses", {"reasoning": {"effort": "low"}, "max_output_tokens": 1}, "reasoning"),
    (google, "google-generative-ai", {"thinkingConfig": {"thinkingBudget": 1}, "maxOutputTokens": 1}, "thinkingConfig"),
])
def test_explicit_reasoning_and_output_override_model_defaults(module, api, params, field):
    model = Model("test", "test", api, "test", "https://example.test", reasoning=True, params=params)
    body = module.build_request(model, context(), SimpleStreamOptions(reasoning="high", max_tokens=4096))
    config = body["generationConfig"] if module is google else body
    assert config[field] != params[field]
    if module is openai_responses:
        assert config["max_output_tokens"] == 4096
    if module is google:
        assert config["maxOutputTokens"] == 4096


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


# ---------------------------------------------------------------------------
# Responses hosted tools
# ---------------------------------------------------------------------------


def hosted_context():
    return TranscriptContext([SystemMessage(content="system", timestamp=0,
        tools_added=[Tool("web_search", "Web search", {}, hosted={"type": "web_search", "search_context_size": "medium"})]),
        UserMessage(content="hello", timestamp=0)])


def test_hosted_tool_declaration_and_raw_item_replay():
    model = Model("test", "test", "openai-responses", "test", "https://example.test")
    body = openai_responses.build_request(model, hosted_context(), SimpleStreamOptions())
    assert body["tools"] == [{"type": "web_search", "search_context_size": "medium"}]

    raw_item = {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "q"}}
    replay = TranscriptContext([*hosted_context().messages,
        assistant([ToolCall(id="ws_1", name="web_search", arguments={}, raw_item=raw_item),
                   TextContent(text="answer")])])
    body = openai_responses.build_request(model, replay, SimpleStreamOptions())
    assert raw_item in body["input"]
    assert not any(isinstance(item, dict) and item.get("type") == "function_call" for item in body["input"])


def test_hosted_tool_sse_capture(monkeypatch):
    payload = [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Searching"},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}},
        {"type": "response.web_search_call.in_progress", "output_index": 1},
        {"type": "response.output_item.done", "output_index": 1, "item": {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "q"}}},
        {"type": "response.completed", "response": {"id": "response", "usage": {"input_tokens": 10, "output_tokens": 5}}},
    ]
    captured = {}
    async def send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, request=request, content="".join("data: " + json.dumps(item) + "\n\n" for item in payload).encode())
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model = Model(id="test", name="test", api="openai-responses", provider="test", base_url="https://example.test", api_key="key")
    async def check():
        return [event async for event in openai_responses.stream_openai_responses(model, hosted_context())]
    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    # Hosted-only output: the loop has nothing to execute.
    assert done.reason == "stop"
    assert done.message.stop_reason == "stop"
    call = next(block for block in done.message.content if isinstance(block, ToolCall))
    assert call.name == "web_search" and call.raw_item["id"] == "ws_1"
    # Captured item replays verbatim on the next request.
    replay = TranscriptContext([*hosted_context().messages, done.message])
    assert openai_responses.build_request(model, replay, SimpleStreamOptions())["input"][-1] == call.raw_item


def test_hosted_and_function_calls_stop_reason(monkeypatch):
    payload = [
        {"type": "response.output_item.done", "output_index": 0, "item": {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "q"}}},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "call_id": "call", "name": "echo", "arguments": ""}},
        {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": '{"text":"hi"}'},
        {"type": "response.completed", "response": {"id": "response", "usage": {"input_tokens": 10, "output_tokens": 5}}},
    ]
    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request, content="".join("data: " + json.dumps(item) + "\n\n" for item in payload).encode())
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model = Model(id="test", name="test", api="openai-responses", provider="test", base_url="https://example.test", api_key="key")
    async def check():
        return [event async for event in openai_responses.stream_openai_responses(model, hosted_context())]
    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert done.reason == "toolUse"


def test_hosted_tools_filtered_from_other_providers():
    system = SystemMessage(content="s", timestamp=0,
        tools_added=[Tool("web_search", "Web search", {}, hosted={"type": "web_search"}),
                     Tool("echo", "Echo", {"type": "object"})])
    ctx = TranscriptContext([system, UserMessage(content="hello", timestamp=0)])
    hosted_call = ToolCall(id="ws_1", name="web_search", arguments={}, raw_item={"type": "web_search_call", "id": "ws_1"})

    claude = Model("test", "test", "anthropic-messages", "test", "https://example.test")
    body = anthropic.build_request(claude, ctx, SimpleStreamOptions())
    assert [t["name"] for t in body["tools"]] == ["echo"]
    assert anthropic.convert_messages([assistant([hosted_call, TextContent(text="a")], api="anthropic-messages")]) == [
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]}]

    body = openai_completions.build_request(Model("test", "test", "openai-completions", "test", "https://example.test"), ctx, SimpleStreamOptions())
    assert body["tools"] == [{"type": "function", "function": {"name": "echo", "description": "Echo", "parameters": {"type": "object"}}}]
    assert "tool_calls" not in openai_completions._assistant_message(assistant([hosted_call]))

    body = google.build_request(Model("test", "test", "google-generative-ai", "test", "https://example.test"), ctx, SimpleStreamOptions())
    assert body["tools"] == [{"functionDeclarations": [{"name": "echo", "description": "Echo", "parametersJsonSchema": {"type": "object"}}]}]
    assert google.build_request(Model("test", "test", "google-generative-ai", "test", "https://example.test"),
        TranscriptContext([assistant([hosted_call])]), SimpleStreamOptions())["contents"] == []

    body = amazon_bedrock.build_request(Model("anthropic.claude-sonnet-4-5", "t", "bedrock-converse-stream", "amazon-bedrock", "https://bedrock"), ctx, SimpleStreamOptions())
    assert body["toolConfig"]["tools"][0]["toolSpec"]["name"] == "echo"
    assert amazon_bedrock.build_request(Model("anthropic.claude-sonnet-4-5", "t", "bedrock-converse-stream", "amazon-bedrock", "https://bedrock"),
        TranscriptContext([assistant([hosted_call])]), SimpleStreamOptions())["messages"] == []


# ---------------------------------------------------------------------------
# Audio / video input
# ---------------------------------------------------------------------------


def media_context(*blocks):
    return TranscriptContext([UserMessage(content=list(blocks), timestamp=0)])


def test_openai_audio_input_parts():
    model = Model("test", "test", "openai-completions", "test", "https://example.test")
    body = openai_completions.build_request(model, media_context(AudioContent(data="AAA", mime_type="audio/wav")), SimpleStreamOptions())
    assert body["messages"][0]["content"] == [{"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}]
    body = openai_completions.build_request(model, media_context(AudioContent(data="AAA", mime_type="audio/mpeg")), SimpleStreamOptions())
    assert body["messages"][0]["content"][0]["input_audio"]["format"] == "mp3"
    with pytest.raises(ValueError, match="audio/wav and audio/mpeg"):
        openai_completions.build_request(model, media_context(AudioContent(data="AAA", mime_type="audio/ogg")), SimpleStreamOptions())

    model = Model("test", "test", "openai-responses", "test", "https://example.test")
    body = openai_responses.build_request(model, media_context(AudioContent(data="AAA", mime_type="audio/wav")), SimpleStreamOptions())
    assert body["input"][0]["content"] == [{"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}]
    with pytest.raises(ValueError, match="video"):
        openai_responses.build_request(model, media_context(VideoContent(data="AAA", mime_type="video/mp4")), SimpleStreamOptions())
    with pytest.raises(ValueError, match="video"):
        openai_completions.build_request(Model("test", "test", "openai-completions", "test", "https://example.test"),
            media_context(VideoContent(data="AAA", mime_type="video/mp4")), SimpleStreamOptions())


def test_google_media_input_parts():
    model = Model("test", "test", "google-generative-ai", "test", "https://example.test")
    body = google.build_request(model, media_context(AudioContent(data="AAA", mime_type="audio/wav"), VideoContent(data="BBB", mime_type="video/mp4")), SimpleStreamOptions())
    assert body["contents"][0]["parts"] == [
        {"inlineData": {"mimeType": "audio/wav", "data": "AAA"}},
        {"inlineData": {"mimeType": "video/mp4", "data": "BBB"}},
    ]


@pytest.mark.parametrize("api", ["anthropic-messages", "bedrock-converse-stream"])
def test_unsupported_media_rejected(api):
    model = Model("test", "test", api, "test", "https://example.test")
    builder = anthropic.build_request if api == "anthropic-messages" else amazon_bedrock.build_request
    with pytest.raises(ValueError, match="audio or video"):
        builder(model, media_context(AudioContent(data="AAA", mime_type="audio/wav")), SimpleStreamOptions())
    with pytest.raises(ValueError, match="audio or video"):
        builder(model, media_context(VideoContent(data="AAA", mime_type="video/mp4")), SimpleStreamOptions())


# ---------------------------------------------------------------------------
# Bedrock per-model inference parameters
# ---------------------------------------------------------------------------


def bedrock_model(model_id="amazon.nova-pro-v1:0", **kwargs):
    return Model(id=model_id, name="t", api="bedrock-converse-stream", provider="amazon-bedrock",
                 base_url="https://bedrock.us-east-1.amazonaws.com", **kwargs)


def test_bedrock_params_merge_and_precedence():
    model = bedrock_model(params={"inferenceConfig": {"topP": 0.9, "temperature": 0.1}, "guardrailConfig": {"guardrailIdentifier": "g1"}})
    body = amazon_bedrock.build_request(model, context(), SimpleStreamOptions())
    # Params extend the built config without dropping maxTokens.
    assert body["inferenceConfig"] == {"maxTokens": body["inferenceConfig"]["maxTokens"], "topP": 0.9, "temperature": 0.1}
    assert body["guardrailConfig"] == {"guardrailIdentifier": "g1"}
    # Explicit caller options win over params.
    body = amazon_bedrock.build_request(model, context(), SimpleStreamOptions(temperature=0.5, max_tokens=4096))
    assert body["inferenceConfig"]["temperature"] == 0.5
    assert body["inferenceConfig"]["maxTokens"] == 4096
    assert body["inferenceConfig"]["topP"] == 0.9


def test_bedrock_top_k_routing_per_family():
    nova = bedrock_model("amazon.nova-pro-v1:0", params={"top_k": 40})
    body = amazon_bedrock.build_request(nova, context(), SimpleStreamOptions())
    assert body["additionalModelRequestFields"] == {"inferenceConfig": {"topK": 40}}
    # A Nova top_k shorthand merges with other model-specific fields.
    nova.params = {"top_k": 40, "additionalModelRequestFields": {"topP": 0.9}}
    body = amazon_bedrock.build_request(nova, context(), SimpleStreamOptions())
    assert body["additionalModelRequestFields"]["inferenceConfig"] == {"topK": 40}
    assert body["additionalModelRequestFields"]["topP"] == 0.9

    claude = bedrock_model("anthropic.claude-sonnet-4-5", params={"top_k": 200})
    body = amazon_bedrock.build_request(claude, context(), SimpleStreamOptions())
    assert body["additionalModelRequestFields"] == {"top_k": 200}


def test_bedrock_thinking_param_precedence():
    claude = bedrock_model("anthropic.claude-sonnet-4-5", reasoning=True,
        params={"additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 4096}, "top_k": 10}})
    # Without an explicit reasoning level, params-provided thinking passes through.
    body = amazon_bedrock.build_request(claude, context(), SimpleStreamOptions())
    assert body["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["additionalModelRequestFields"]["top_k"] == 10
    # With reasoning set, the derived budget overrides only the thinking key.
    body = amazon_bedrock.build_request(claude, context(), SimpleStreamOptions(reasoning="high", max_tokens=32768))
    assert body["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 16384
    assert body["additionalModelRequestFields"]["top_k"] == 10
    assert "temperature" not in body["inferenceConfig"]


# ---------------------------------------------------------------------------
# JSON round-trip for the new block and tool fields
# ---------------------------------------------------------------------------


def test_json_round_trip_new_blocks():
    message = assistant([
        ToolCall(id="ws_1", name="web_search", arguments={}, raw_item={"type": "web_search_call", "id": "ws_1"}),
        ToolCall(id="call", name="echo", arguments={"text": "hi"}, thought_signature="sig"),
    ])
    restored = message_from_json(to_json(message))
    assert restored.content[0].raw_item == {"type": "web_search_call", "id": "ws_1"}
    assert restored.content[1].thought_signature == "sig"
    assert restored.content[1].raw_item is None

    user = UserMessage(content=[TextContent(text="look"), ImageContent(data="A", mime_type="image/png"),
        AudioContent(data="B", mime_type="audio/wav"), VideoContent(data="C", mime_type="video/mp4")], timestamp=0)
    restored = message_from_json(to_json(user))
    assert restored.content[2] == AudioContent(data="B", mime_type="audio/wav")
    assert restored.content[3] == VideoContent(data="C", mime_type="video/mp4")

    system = SystemMessage(content="", timestamp=0, tools_added=[Tool("web_search", "Web", {}, hosted={"type": "web_search"})])
    restored = message_from_json(to_json(system))
    assert restored.tools_added[0].hosted == {"type": "web_search"}
