"""Chat Completions token limits and optional usage streaming."""

import pytest

from pi.ai.models import Model, resolve_model
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.providers.openai_completions import build_request
from pi.ai.types import TranscriptContext, UserMessage
from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.model_registry import ModelRegistry


@pytest.mark.parametrize("limit", [None, 123])
def test_builtin_gpt51_uses_completion_tokens(limit):
    model = resolve_model("openai", "gpt-5.1", api_key="test")
    body = build_request(model, TranscriptContext([UserMessage(content="hi", timestamp=0)]),
                         SimpleStreamOptions(max_tokens=limit))
    assert body["max_completion_tokens"] == (limit or model.max_tokens)
    assert "max_tokens" not in body
    assert body["stream_options"] == {"include_usage": True}


def test_custom_endpoint_keeps_max_tokens():
    model = Model("local", "local", "openai-completions", "local", "http://localhost/v1")
    body = build_request(model, TranscriptContext([]), None)
    assert body["max_tokens"] == model.max_tokens
    assert "max_completion_tokens" not in body


def test_registry_compat_controls_request(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "agent"))
    write_json(config.agent_dir / "models.json", {"providers": {"openai": {"models": [
        {"id": "gpt-5.1", "compat": {"maxTokensField": "max_tokens", "supportsUsageInStreaming": False}},
    ]}}})
    model = ModelRegistry(config).resolve("openai", "gpt-5.1", api_key="test")
    body = build_request(model, TranscriptContext([]), SimpleStreamOptions(max_tokens=42))
    assert body["max_tokens"] == 42
    assert "max_completion_tokens" not in body
    assert "stream_options" not in body
