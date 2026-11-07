import asyncio
import json

import httpx
import pytest

from pi.ai.json_io import message_from_json, to_json
from pi.ai.providers import get_stream_fn
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import TranscriptContext, UserMessage, SystemMessage, Tool
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry


def sse(events):
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()


def radius_model(tmp_path):
    return ModelRegistry(Configuration(tmp_path, tmp_path / "agent")).resolve("radius", "test-model", api_key="key")


@pytest.mark.parametrize("terminal", ["done", "error", "missing"])
def test_radius_protocol(tmp_path, monkeypatch, terminal):
    model = ModelRegistry(Configuration(tmp_path, tmp_path / "agent")).resolve("radius", "test-model", api_key="key")
    captured = {}
    events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "hello"},
        {"type": "text_end", "contentIndex": 0, "content": "hello"},
        {"type": "thinking_start", "contentIndex": 1},
        {"type": "thinking_end", "contentIndex": 1, "content": "thought", "contentSignature": "signed"},
        {"type": "toolcall_start", "contentIndex": 2, "id": "c", "toolName": "echo"},
        {"type": "toolcall_delta", "contentIndex": 2, "delta": '{"x":1}'},
        {"type": "toolcall_end", "contentIndex": 2, "toolCall": {"type": "toolCall", "id": "c", "name": "echo", "arguments": {"x": 1}}},
    ]
    if terminal != "missing":
        events.append({"type": terminal, "reason": "toolUse" if terminal == "done" else "error",
                       "errorMessage": "denied", "responseId": "response", "usage": {"input": 5, "cacheRead": 3, "cost": {"total": 0.2}}})
    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content="".join("data: " + json.dumps(e) + "\n\n" for e in events).encode())
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([SystemMessage("sys", 0, tools_added=[Tool("echo", "Echo", {"type": "object"})]), UserMessage("hi", 0)])
    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key", reasoning="high"))]
    result = asyncio.run(check())
    assert sum(e.type in ("done", "error") for e in result) == 1
    last = result[-1]
    message = last.message if terminal == "done" else last.error
    assert message.content[0].text == "hello"
    assert message.content[1].thinking_signature == "signed"
    assert message.content[2].arguments == {"x": 1}
    if terminal != "missing":
        assert message.usage.cache_read == 3 and message.usage.cost["total"] == 0.2
        assert message.response_id == "response"
    request = captured["request"]
    assert request.url.path == "/messages"
    body = json.loads(request.content)
    assert body["context"]["messages"][0]["toolsAdded"][0]["name"] == "echo"
    assert body["options"]["reasoning"] == "high"


@pytest.mark.parametrize("entra", [False, True])
def test_azure_responses(tmp_path, monkeypatch, entra):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "v1")
    if entra:
        monkeypatch.setenv("AZURE_ACCESS_TOKEN", "token")
    model = ModelRegistry(Configuration(tmp_path, tmp_path / "agent")).resolve("azure-openai-responses")
    model.params = {"deployment_name": "my-deployment"}
    captured = {}
    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=b'data: {"type":"response.output_text.delta","delta":"hi"}\n\ndata: {"type":"response.completed","response":{"usage":{}}}\n\n')
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async def check():
        key = await model.get_api_key(model.provider)
        return [e async for e in get_stream_fn(model.api)(model, TranscriptContext([UserMessage("hi", 0)]), SimpleStreamOptions(api_key=key))]
    result = asyncio.run(check())
    assert result[-1].type == "done"
    assert result[-1].message.content[0].text == "hi"
    request = captured["request"]
    assert str(request.url) == "https://resource.openai.azure.com/openai/v1/responses?api-version=v1"
    assert request.headers["authorization" if entra else "api-key"] == ("Bearer token" if entra else "key")
    body = json.loads(request.content)
    assert body["model"] == "my-deployment" and "deployment_name" not in body


def test_radius_text_signature_roundtrip(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_end", "contentIndex": 0, "content": "hello", "contentSignature": "tsig"},
        {"type": "done", "reason": "stop", "usage": {"input": 1, "output": 1}},
    ]
    captured = {}

    async def send(self, request, **kwargs):
        captured.setdefault("requests", []).append(request)
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    def collect(context):
        async def check():
            return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]
        return asyncio.run(check())

    message = collect(TranscriptContext([UserMessage("hi", 0)]))[-1].message
    assert message.content[0].text_signature == "tsig"

    # The assistant message replays into the next request with its signature.
    collect(TranscriptContext([UserMessage("hi", 0), message]))
    body = json.loads(captured["requests"][1].content)
    assert body["context"]["messages"][1]["content"][0]["textSignature"] == "tsig"


@pytest.mark.parametrize("terminal", ["done", "error"])
def test_radius_provider_thinking_level(tmp_path, monkeypatch, terminal):
    model = radius_model(tmp_path)
    events = [
        {"type": "start"},
        {"type": "thinking_start", "contentIndex": 0},
        {"type": "thinking_end", "contentIndex": 0, "content": "hmm", "contentSignature": "sig"},
        {"type": terminal, "reason": "stop" if terminal == "done" else "error",
         "usage": {"input": 1}, "providerThinkingLevel": "high"},
    ]

    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]

    result = asyncio.run(check())
    message = result[-1].message if terminal == "done" else result[-1].error
    assert message.provider_thinking_level == "high"
    restored = message_from_json(to_json(message))
    assert restored.provider_thinking_level == "high"


def test_radius_rewrite_diagnostic(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    rewrite = {"policyId": "policy", "policyVersion": 3, "changed": True, "tokenCountChange": -5,
               "messageCountChange": 0, "systemPromptChanged": False}
    events = [{"type": "start"}, {"type": "done", "reason": "stop", "usage": {"input": 1}, "rewrite": rewrite}]

    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]

    result = asyncio.run(check())
    message = result[-1].message
    diagnostic = message.diagnostics[0]
    assert diagnostic.type == "pi_messages_rewrite"
    assert diagnostic.details == rewrite
    assert diagnostic.error is None
    # Diagnostics serialize with the message for session storage and the wire.
    data = to_json(message)
    assert data["diagnostics"][0]["type"] == "pi_messages_rewrite"
    assert data["diagnostics"][0]["details"] == rewrite


def test_radius_response_failure_diagnostic(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    captured = {}

    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(401, request=request,
                              content=json.dumps({"error": {"message": "Token expired", "code": "unauthorized"}}).encode())

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]

    result = asyncio.run(check())
    error = result[-1].error
    assert result[-1].type == "error"
    assert "401" in error.error_message and "Token expired" in error.error_message and "unauthorized" in error.error_message
    diagnostic = error.diagnostics[0]
    assert diagnostic.type == "pi_messages_response_failure"
    assert diagnostic.details["version"] == 1
    assert diagnostic.details["status"] == 401
    assert diagnostic.details["statusText"] == "Unauthorized"
    assert diagnostic.details["error"] == {"message": "Token expired", "code": "unauthorized"}
    assert diagnostic.error.code == "unauthorized"


def test_radius_response_failure_plain_body(tmp_path, monkeypatch):
    model = radius_model(tmp_path)

    async def send(self, request, **kwargs):
        return httpx.Response(500, request=request, content=b"x" * 9000)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]

    result = asyncio.run(check())
    error = result[-1].error
    assert error.error_message.startswith("500")
    diagnostic = error.diagnostics[0]
    assert diagnostic.details["status"] == 500
    # The raw body is embedded only after the 8192-character truncation.
    assert len(diagnostic.details["body"]) == 8193
    assert diagnostic.details["body"].endswith("…")
    assert "error" not in diagnostic.details


def test_radius_request_options(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    captured = {}
    events = [{"type": "start"}, {"type": "done", "reason": "stop", "usage": {"input": 1}}]

    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setenv("PI_CACHE_RETENTION", "long")
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        options = SimpleStreamOptions(api_key="key", max_tokens=100, session_id="s1", tool_choice="auto", debug=True)
        return [e async for e in get_stream_fn(model.api)(model, context, options)]

    asyncio.run(check())
    request = captured["request"]
    assert str(request.url).endswith("/messages?debug=1")
    body = json.loads(request.content)
    assert body["options"] == {"maxTokens": 100, "cacheRetention": "long", "sessionId": "s1", "toolChoice": "auto"}


def test_radius_cache_retention_mapping(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    captured = {}
    events = [{"type": "start"}, {"type": "done", "reason": "stop", "usage": {"input": 1}}]

    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check(retention):
        options = SimpleStreamOptions(api_key="key", cache_retention=retention)
        return [e async for e in get_stream_fn(model.api)(model, context, options)]

    asyncio.run(check("5m"))
    assert json.loads(captured["request"].content)["options"]["cacheRetention"] == "short"
    asyncio.run(check("1h"))
    assert json.loads(captured["request"].content)["options"]["cacheRetention"] == "long"
    asyncio.run(check(None))
    assert "cacheRetention" not in json.loads(captured["request"].content)["options"]


def test_radius_unknown_event_forwarded(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "gateway.custom", "detail": "upstream"},
        {"type": "done", "reason": "stop", "usage": {"input": 1}},
    ]

    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        return [e async for e in get_stream_fn(model.api)(model, context, SimpleStreamOptions(api_key="key"))]

    result = asyncio.run(check())
    assert sum(getattr(e, "type", None) in ("done", "error") for e in result) == 1
    forwarded = [e for e in result if isinstance(e, dict)]
    assert forwarded and forwarded[0]["type"] == "gateway.custom"
    assert forwarded[0]["partial"].content[0].text == ""


def test_radius_request_hooks(tmp_path, monkeypatch):
    model = radius_model(tmp_path)
    captured = {}
    events = [
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "hi"},
        {"type": "done", "reason": "stop", "usage": {"input": 1}},
    ]

    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=sse(events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    raw_events = []
    responses = []

    async def on_response(response):
        responses.append(response)

    async def on_stream_event(event):
        raw_events.append(event)

    def on_payload(payload, payload_model):
        payload["options"]["injected"] = True
        return payload

    context = TranscriptContext([UserMessage("hi", 0)])

    async def check():
        options = SimpleStreamOptions(api_key="key", on_payload=on_payload, on_response=on_response,
                                      on_provider_stream_event=on_stream_event)
        return [e async for e in get_stream_fn(model.api)(model, context, options)]

    result = asyncio.run(check())
    assert result[-1].type == "done"
    assert json.loads(captured["request"].content)["options"]["injected"] is True
    assert responses[0]["status"] == 200
    assert isinstance(responses[0]["headers"], dict)
    assert [event["type"] for event in raw_events] == ["start", "text_start", "text_delta", "done"]
