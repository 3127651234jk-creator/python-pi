import asyncio
import json

import httpx
import pytest

from pi.ai.providers import get_stream_fn
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import TranscriptContext, UserMessage, SystemMessage, Tool
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry


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
