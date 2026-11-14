import json

import pytest

from pi.ai.models import Model
from pi.ai.providers import amazon_bedrock, google, openai_responses
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, TranscriptContext, Usage


@pytest.mark.parametrize("module,api", [
    (amazon_bedrock, "bedrock-converse-stream"),
    (google, "google-generative-ai"),
    (openai_responses, "openai-responses"),
])
@pytest.mark.parametrize("changed", [None, "model", "provider", "api"])
def test_signatures_only_replay_to_origin(module, api, changed):
    model = Model("origin", "Origin", api, "provider", "https://example.test")
    signature = json.dumps({"type": "reasoning", "encrypted_content": "opaque-signature"})
    message = AssistantMessage(
        content=[ThinkingContent(thinking="visible reasoning", thinking_signature=signature),
                 TextContent(text="answer"),
                 ToolCall(id="call", name="echo", arguments={}, thought_signature="tool-signature")],
        api=api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=0,
    )
    if changed:
        setattr(message, changed, "other")
    body = module.build_request(model, TranscriptContext([message]), SimpleStreamOptions())
    encoded = json.dumps(body)
    assert ("opaque-signature" in encoded) == (changed is None)
    if module is google:
        assert ("tool-signature" in encoded) == (changed is None)
    assert "visible reasoning" in encoded or (module is openai_responses and changed is None)
    assert "answer" in encoded
    assert message.content[0].thinking_signature == signature


@pytest.mark.parametrize("module,api", [
    (amazon_bedrock, "bedrock-converse-stream"),
    (google, "google-generative-ai"),
    (openai_responses, "openai-responses"),
])
def test_foreign_redacted_reasoning_is_not_replayed(module, api):
    model = Model("target", "Target", api, "provider", "https://example.test")
    message = AssistantMessage(
        content=[ThinkingContent(thinking="opaque", thinking_signature="invalid-json", redacted=True),
                 TextContent(text="answer")],
        api=api, provider=model.provider, model="origin", usage=Usage(), stop_reason="stop", timestamp=0,
    )
    body = module.build_request(model, TranscriptContext([message]), SimpleStreamOptions())
    assert "opaque" not in json.dumps(body)
    assert "answer" in json.dumps(body)
