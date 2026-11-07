import asyncio
import json
import time

import httpx
import pytest

from pi.ai.providers import get_stream_fn
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import TranscriptContext, UserMessage
from pi.coding_agent import radius_catalog
from pi.coding_agent.config import Configuration, read_json, write_json
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.radius_catalog import (fetch_radius_gateway_config, load_cache, normalize_gateway_config,
                                            radius_overlay, refresh_radius_catalog, save_cache)
from pi.coding_agent.cli import refresh_radius_models

GATEWAY_CONFIG = {
    "baseUrl": "https://gw.example/v1",
    "models": [
        {"id": "balanced", "name": "Balanced", "reasoning": True,
         "thinkingLevelMap": {"off": None, "low": "low", "high": "high"},
         "input": ["text", "image"],
         "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0},
         "contextWindow": 1048576, "maxTokens": 131072},
        {"id": "max", "name": "Max", "reasoning": False, "input": ["text"],
         "cost": {"input": 1, "output": 2}, "contextWindow": 200000, "maxTokens": 8192},
        {"id": "broken", "reasoning": True},
    ],
}


def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(Configuration(tmp_path, tmp_path / "agent"))


def stub_config_response(monkeypatch, captured, body=None, status=200):
    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(status, request=request,
                              content=body if body is not None else json.dumps(GATEWAY_CONFIG).encode())

    monkeypatch.setattr(httpx.AsyncClient, "send", send)


def test_fetch_radius_gateway_config(monkeypatch):
    captured = {}
    stub_config_response(monkeypatch, captured)
    config = asyncio.run(fetch_radius_gateway_config("gw.example", api_key="key"))
    request = captured["request"]
    assert str(request.url) == "https://gw.example/v1/config"
    assert request.headers["accept"] == "application/json"
    assert request.headers["authorization"] == "Bearer key"
    assert config["base_url"] == "https://gw.example/v1"
    assert [model["id"] for model in config["models"]] == ["balanced", "max"]
    balanced = config["models"][0]
    assert balanced["costs"] == {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 0}
    assert balanced["base_url"] == "https://gw.example/v1"
    assert balanced["thinking_level_map"] == {"off": None, "low": "low", "high": "high"}
    assert balanced["input"] == ["text", "image"]
    assert balanced["context_window"] == 1048576 and balanced["max_tokens"] == 131072


def test_fetch_radius_gateway_config_unauthenticated(monkeypatch):
    captured = {}
    stub_config_response(monkeypatch, captured)
    asyncio.run(fetch_radius_gateway_config("https://gw.example/"))
    assert "authorization" not in captured["request"].headers
    assert str(captured["request"].url) == "https://gw.example/v1/config"


def test_fetch_radius_gateway_config_errors(monkeypatch):
    captured = {}
    stub_config_response(monkeypatch, captured, body=b" gateway down \n", status=503)
    with pytest.raises(ValueError) as exc:
        asyncio.run(fetch_radius_gateway_config("gw.example"))
    assert "Could not load Radius config from https://gw.example: 503: gateway down" in str(exc.value)

    stub_config_response(monkeypatch, captured, body=b'{"nope": 1}')
    with pytest.raises(ValueError) as exc:
        asyncio.run(fetch_radius_gateway_config("gw.example"))
    assert "Invalid Radius config from https://gw.example" in str(exc.value)


def test_normalize_gateway_config_input_fallback():
    config = normalize_gateway_config({"baseUrl": "https://gw.example/v1", "models": [
        {"id": "m", "name": "M", "reasoning": False, "input": ["image"], "cost": {},
         "contextWindow": 100, "maxTokens": 50}]}, "gw.example")
    assert config["models"][0]["input"] == ["text", "image"]
    assert "costs" not in config["models"][0]


def test_registry_overlay_and_stream_endpoint(tmp_path, monkeypatch):
    save_cache(tmp_path / "agent", "https://gw.example", normalize_gateway_config(GATEWAY_CONFIG, "https://gw.example"))
    reg = registry(tmp_path)
    assert "radius/balanced" in reg.models
    model = reg.resolve("radius", "balanced", api_key="key")
    assert model.api == "pi-messages"
    assert model.base_url == "https://gw.example/v1"
    assert model.context_window == 1048576 and model.max_tokens == 131072
    assert model.reasoning is True
    assert model.costs.input == 3 and model.costs.cache_read == 0.3
    assert model.thinking_level_map == {"off": None, "low": "low", "high": "high"}
    assert reg.resolve("radius", "max").reasoning is False

    # Streaming goes to the gateway-advertised baseUrl, not the provider default.
    captured = {}

    async def send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request,
                              content=b'data: {"type":"done","reason":"stop","usage":{"input":1}}\n\n')

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    async def check():
        options = SimpleStreamOptions(api_key="key")
        stream = get_stream_fn(model.api)(model, TranscriptContext([UserMessage("hi", 0)]), options)
        return [e async for e in stream]

    asyncio.run(check())
    assert str(captured["request"].url) == "https://gw.example/v1/messages"


def test_user_models_json_models_win(tmp_path):
    agent_dir = tmp_path / "agent"
    save_cache(agent_dir, "https://gw.example", normalize_gateway_config(GATEWAY_CONFIG, "https://gw.example"))
    write_json(agent_dir / "models.json",
               {"providers": {"radius": {"models": [{"id": "balanced", "context_window": 123}]}}})
    reg = ModelRegistry(Configuration(tmp_path, agent_dir))
    assert reg.resolve("radius", "balanced").context_window == 123
    # The user did not declare "max"; discovery still provides it.
    assert reg.resolve("radius", "max").context_window == 200000


def test_radius_overlay_gating(tmp_path):
    agent_dir = tmp_path / "agent"
    save_cache(agent_dir, "https://gw.example", normalize_gateway_config(GATEWAY_CONFIG, "https://gw.example"))
    assert radius_overlay(agent_dir, enabled=False) == {}
    overlay = radius_overlay(agent_dir)
    assert list(overlay) == ["radius"]
    assert [model["id"] for model in overlay["radius"]["models"]] == ["balanced", "max"]


def test_radius_cache_expiry(tmp_path):
    agent_dir = tmp_path / "agent"
    save_cache(agent_dir, "https://gw.example", normalize_gateway_config(GATEWAY_CONFIG, "https://gw.example"))
    assert load_cache(agent_dir) is not None
    data = read_json(radius_catalog.cache_path(agent_dir))
    data["fetched_at"] = time.time() - radius_catalog.CACHE_MAX_AGE - 1
    write_json(radius_catalog.cache_path(agent_dir), data)
    assert load_cache(agent_dir) is None


def test_refresh_radius_catalog(monkeypatch, tmp_path):
    captured = {}
    stub_config_response(monkeypatch, captured)
    monkeypatch.setenv("RADIUS_API_KEY", "env-key")
    reg = registry(tmp_path)
    count = asyncio.run(refresh_radius_catalog(reg))
    assert count == 2
    assert captured["request"].headers["authorization"] == "Bearer env-key"
    assert "radius/balanced" in reg.models


def test_refresh_radius_catalog_credential_gateway(monkeypatch, tmp_path):
    captured = {}
    stub_config_response(monkeypatch, captured)
    monkeypatch.delenv("RADIUS_GATEWAY", raising=False)
    monkeypatch.delenv("RADIUS_API_KEY", raising=False)
    reg = registry(tmp_path)
    reg.auth.set("radius", {"type": "api_key", "key": "stored", "gateway": "https://custom.example"})
    asyncio.run(refresh_radius_catalog(reg))
    assert str(captured["request"].url) == "https://custom.example/v1/config"
    assert captured["request"].headers["authorization"] == "Bearer stored"


def test_startup_discovery_skips_without_radius_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("RADIUS_API_KEY", raising=False)
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    reg = registry(tmp_path)

    async def send(self, request, **kwargs):
        raise AssertionError("startup must not fetch the Radius gateway without credentials")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    asyncio.run(refresh_radius_models(reg))
    assert [model for model in reg.models.values() if model.provider == "radius"] == []


def test_startup_discovery_fetches_with_radius_env_key(monkeypatch, tmp_path):
    captured = {}
    stub_config_response(monkeypatch, captured)
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    monkeypatch.setenv("RADIUS_API_KEY", "env-key")
    reg = registry(tmp_path)
    asyncio.run(refresh_radius_models(reg))
    assert "radius/balanced" in reg.models
    assert captured["request"].headers["authorization"] == "Bearer env-key"


def test_refresh_radius_catalog_env_gateway(monkeypatch, tmp_path):
    captured = {}
    stub_config_response(monkeypatch, captured)
    monkeypatch.setenv("RADIUS_GATEWAY", "https://env.example")
    monkeypatch.delenv("RADIUS_API_KEY", raising=False)
    reg = registry(tmp_path)
    asyncio.run(refresh_radius_catalog(reg))
    assert str(captured["request"].url) == "https://env.example/v1/config"
    assert "authorization" not in captured["request"].headers
