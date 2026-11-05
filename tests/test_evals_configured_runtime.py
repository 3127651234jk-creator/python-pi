"""Port of packages/evals/test/configured-runtime.test.ts."""

import asyncio
import json

import httpx
import pytest

from evals.acme_server import (
    OPENAI_MODEL_ID,
    OPENAI_PROBE_PROMPT,
    OPENAI_PROBE_RESPONSE,
    OPENAI_PROVIDER_ID,
    create_acme_server,
)
from evals.configured_runtime import (
    ProviderScenario,
    inspect_added_model,
    inspect_provider,
    load_configured_model_runtime,
)
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import TranscriptContext, UserMessage

PROVIDER_ID = OPENAI_PROVIDER_ID
MODEL = {
    "id": OPENAI_MODEL_ID,
    "name": "Acme Chat",
    "provider": PROVIDER_ID,
    "reasoning": False,
    "input": ["text"],
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 32768,
    "maxTokens": 4096,
}


@pytest.fixture(autouse=True)
def _direct_httpx(direct_httpx):
    """Opt every probe in this module into direct localhost HTTP."""
    del direct_httpx


@pytest.fixture(scope="module")
def server():
    acme = create_acme_server("openai")
    acme.start()
    yield acme
    acme.stop()


@pytest.fixture(autouse=True)
def reset_probe(server):
    server.reset()


def agent_dir_with(tmp_path, models_json):
    directory = tmp_path / "agent"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "models.json").write_text(json.dumps(models_json) + "\n", encoding="utf-8")
    return directory


def runtime_with(tmp_path, models_json):
    return load_configured_model_runtime(agent_dir_with(tmp_path, models_json))


def acme_models_json(server):
    return {"providers": {PROVIDER_ID: {
        "base_url": server.base_url(),
        "api": "openai-completions",
        "api_key": "$ACME_API_KEY",
        "models": [{
            "id": MODEL["id"],
            "name": MODEL["name"],
            "reasoning": MODEL["reasoning"],
            "input": MODEL["input"],
            "costs": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            "context_window": MODEL["contextWindow"],
            "max_tokens": MODEL["maxTokens"],
        }],
    }}}


def probe(server, content, api_key):
    return ProviderScenario(
        provider_id=PROVIDER_ID,
        model_id=OPENAI_MODEL_ID,
        create_context=lambda: TranscriptContext([UserMessage(content=content, timestamp=0)]),
        options=SimpleStreamOptions(max_tokens=32),
        valid_request_received=server.valid_request_received,
    )


class TestInspectProvider:
    def test_probes_a_models_json_provider_through_the_registry(self, tmp_path, monkeypatch, server):
        monkeypatch.setenv("ACME_API_KEY", "resolved-acme-key")
        registry = runtime_with(tmp_path, acme_models_json(server))
        result = asyncio.run(inspect_provider(registry, probe(server, OPENAI_PROBE_PROMPT, "resolved-acme-key")))
        assert result == {"result": {
            "validRequestReceived": True,
            "model": MODEL,
            "response": {"text": OPENAI_PROBE_RESPONSE, "stopReason": "stop", "inputTokens": 3, "outputTokens": 2},
        }}

    def test_keeps_a_non_probe_completion_distinct_from_a_valid_probe(self, tmp_path, monkeypatch, server):
        monkeypatch.setenv("ACME_API_KEY", "resolved-acme-key")
        registry = runtime_with(tmp_path, acme_models_json(server))
        result = asyncio.run(inspect_provider(registry, probe(server, "hello", "resolved-acme-key")))
        assert result == {"result": {
            "validRequestReceived": False,
            "model": MODEL,
            "response": {"text": OPENAI_PROBE_RESPONSE, "stopReason": "stop", "inputTokens": 3, "outputTokens": 2},
        }}

    def test_returns_a_structured_error_when_the_configured_model_is_missing(self, tmp_path, monkeypatch, server):
        monkeypatch.setenv("ACME_API_KEY", "resolved-acme-key")
        registry = runtime_with(tmp_path, {"providers": {}})
        result = asyncio.run(inspect_provider(registry, probe(server, OPENAI_PROBE_PROMPT, "resolved-acme-key")))
        assert result == {"result": {"error": f"Model {PROVIDER_ID}/{OPENAI_MODEL_ID} is unavailable after reload."}}
        assert server.valid_request_received() is False

    def test_returns_a_structured_error_when_models_json_cannot_be_parsed(self, tmp_path, monkeypatch, server):
        monkeypatch.setenv("ACME_API_KEY", "resolved-acme-key")
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "models.json").write_text("{", encoding="utf-8")
        registry = load_configured_model_runtime(directory)
        result = asyncio.run(inspect_provider(registry, probe(server, OPENAI_PROBE_PROMPT, "resolved-acme-key")))
        assert "Cannot read configuration" in result["result"]["error"]
        assert server.valid_request_received() is False

    def test_does_not_treat_an_unauthorized_completion_as_a_valid_probe(self, tmp_path, monkeypatch, server):
        monkeypatch.setenv("ACME_API_KEY", "wrong-key")
        registry = runtime_with(tmp_path, acme_models_json(server))
        result = asyncio.run(inspect_provider(registry, probe(server, OPENAI_PROBE_PROMPT, "wrong-key")))
        assert result == {"result": {
            "validRequestReceived": False,
            "model": MODEL,
            "response": {"text": "", "stopReason": "error", "inputTokens": 0, "outputTokens": 0},
        }}


class TestInspectAddedModel:
    PROVIDER = "openai"
    MODEL_ID = "fixture-chat"
    ADDED_MODEL = {
        "id": MODEL_ID,
        "name": "Fixture Chat",
        "provider": PROVIDER,
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 32768,
        "maxTokens": 4096,
    }

    def added_models_json(self):
        return {"providers": {self.PROVIDER: {
            # The Python registry validates every provider endpoint, so the
            # built-in base URL is declared explicitly (TS inherits it).
            "base_url": "https://api.openai.com/v1",
            "models": [{
                "id": self.ADDED_MODEL["id"],
                "name": self.ADDED_MODEL["name"],
                "reasoning": self.ADDED_MODEL["reasoning"],
                "input": self.ADDED_MODEL["input"],
                "costs": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "context_window": self.ADDED_MODEL["contextWindow"],
                "max_tokens": self.ADDED_MODEL["maxTokens"],
            }],
        }}}

    def test_loads_an_added_model_without_dropping_builtin_models(self, tmp_path):
        registry = runtime_with(tmp_path, self.added_models_json())
        result = asyncio.run(inspect_added_model(registry, self.PROVIDER, self.MODEL_ID))
        assert result == {"result": {"model": self.ADDED_MODEL, "existingModelsPreserved": True}}

    def test_returns_a_structured_error_when_the_added_model_is_missing(self, tmp_path):
        registry = runtime_with(tmp_path, {"providers": {}})
        result = asyncio.run(inspect_added_model(registry, self.PROVIDER, self.MODEL_ID))
        assert result == {"result": {"error": f"Model {self.PROVIDER}/{self.MODEL_ID} is unavailable after reload."}}

    def test_returns_a_structured_error_when_models_json_cannot_be_parsed(self, tmp_path):
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "models.json").write_text("{", encoding="utf-8")
        registry = load_configured_model_runtime(directory)
        result = asyncio.run(inspect_added_model(registry, self.PROVIDER, self.MODEL_ID))
        assert "Cannot read configuration" in result["result"]["error"]
