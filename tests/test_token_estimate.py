"""Context token estimation and the output-limit clamp (B-AI-03, #10497)."""

import json
import math

import pytest

from pi.ai.estimate import (
    CHARS_PER_TOKEN,
    ESTIMATED_MEDIA_CHARS,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
)
from pi.ai.models import Model
from pi.ai.providers import amazon_bedrock, anthropic, google, openai_completions, openai_responses
from pi.ai.providers.base import SimpleStreamOptions, clamp_max_tokens_to_context, resolve_max_tokens
from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    TranscriptContext,
    Usage,
    UserMessage,
)


def assistant(content, usage=None, stop_reason="stop", timestamp=0, **kwargs):
    return AssistantMessage(
        content=content, api="test", provider="test", model="test",
        usage=usage or Usage(), stop_reason=stop_reason, timestamp=timestamp, **kwargs,
    )


def usage(input=0, output=0, cache_read=0, cache_write=0):
    return Usage(input=input, output=output, cache_read=cache_read, cache_write=cache_write)


# ---------------------------------------------------------------------------
# Shared estimator: 3.5 characters per token
# ---------------------------------------------------------------------------

def test_estimate_text_tokens_uses_3_5_chars_per_token():
    assert CHARS_PER_TOKEN == 3.5
    assert estimate_text_tokens("xxxxxxx") == 2  # ceil(7 / 3.5)
    assert estimate_text_tokens("") == 0


def test_estimate_message_tokens_counts_content_not_envelope():
    assert estimate_message_tokens(UserMessage(content="x" * 350, timestamp=0)) == 100
    # Media blocks count as a fixed char budget, never their base64 payload.
    media = UserMessage(
        content=[TextContent(text="hi"), ImageContent(data="A" * 90000, mime_type="image/png")], timestamp=0
    )
    assert estimate_message_tokens(media) == math.ceil((2 + ESTIMATED_MEDIA_CHARS) / CHARS_PER_TOKEN)
    # Assistant: text + thinking as text, tool calls as name + arguments JSON.
    arguments = {"text": "c" * 35}
    call = assistant([
        TextContent(text="a" * 35),
        ThinkingContent(thinking="b" * 70),
        ToolCall(id="call", name="echo", arguments=arguments),
    ])
    expected_chars = 35 + 70 + len("echo") + len(json.dumps(arguments))
    assert estimate_message_tokens(call) == math.ceil(expected_chars / CHARS_PER_TOKEN)


def test_estimate_message_tokens_system_counts_prompt_and_tools():
    bare = SystemMessage(content="p" * 350, timestamp=0)
    assert estimate_message_tokens(bare) == 100
    # Sections count as prompt text.
    sectioned = SystemMessage(content="", timestamp=0, sections={"guide": "s" * 35})
    assert estimate_message_tokens(sectioned) == 10
    # Tool declarations add tokens beyond the prompt text.
    declared = SystemMessage(
        content="p" * 350, timestamp=0,
        tools_added=[Tool("echo", "Echo", {"type": "object"})],
        tools_removed=[ToolReference(name="gone")],
    )
    assert estimate_message_tokens(declared) > 100


# ---------------------------------------------------------------------------
# Context estimate: usage anchoring
# ---------------------------------------------------------------------------

def test_calculate_context_tokens_sums_usage_components():
    assert calculate_context_tokens(usage(input=10, output=5, cache_read=2)) == 17


def test_estimate_context_tokens_anchors_on_last_assistant_usage():
    messages = [
        UserMessage(content="x" * 3500, timestamp=0),
        assistant([TextContent(text="reply")], usage=usage(input=900, output=100, cache_read=50), timestamp=100),
        UserMessage(content="y" * 350, timestamp=200),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.usage_tokens == 1050
    assert estimate.trailing_tokens == 100  # the trailing user message
    assert estimate.tokens == 1150
    assert estimate.last_usage_index == 1


def test_estimate_context_tokens_skips_error_and_aborted_usage():
    messages = [
        assistant([TextContent(text="boom")], usage=usage(input=999, output=999), stop_reason="error", timestamp=0),
        assistant([TextContent(text="cut")], usage=usage(input=999, output=999), stop_reason="aborted", timestamp=1),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0
    assert estimate.tokens == estimate.trailing_tokens > 0


def test_estimate_context_tokens_ignores_usage_under_a_newer_prefix():
    """A compaction summary inserted with a newer timestamp invalidates the usage."""
    messages = [
        UserMessage(content="summary inserted after the fact", timestamp=1000),
        assistant([TextContent(text="old reply")], usage=usage(input=999, output=999), timestamp=500),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_without_usage_estimates_everything():
    messages = [UserMessage(content="z" * 700, timestamp=0), assistant([TextContent(text="w" * 35)], timestamp=1)]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens == estimate.trailing_tokens == 210
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_accepts_transcript_context():
    context = TranscriptContext([UserMessage(content="x" * 350, timestamp=0)])
    assert estimate_context_tokens(context).tokens == 100


# ---------------------------------------------------------------------------
# Output-limit clamp
# ---------------------------------------------------------------------------

def small_model(api="anthropic-messages", context_window=2000, **kwargs):
    return Model(id="m", name="m", api=api, provider="test", base_url="https://example.test",
                 context_window=context_window, max_tokens=8192, **kwargs)


def test_clamp_floors_at_one_token_when_nothing_fits():
    model = small_model()
    context = TranscriptContext([UserMessage(content="x" * 3500, timestamp=0)])  # 1000 tokens
    # 2000 - 1000 - 4096 is negative: nothing left, floor at one token.
    assert clamp_max_tokens_to_context(model, context, 8192) == 1
    assert resolve_max_tokens(model, context, SimpleStreamOptions(max_tokens=8192)) == 1


def test_clamp_limits_to_available_window():
    model = Model(id="m", name="m", api="anthropic-messages", provider="test",
                  base_url="https://example.test", context_window=50_000, max_tokens=8192)
    context = TranscriptContext([UserMessage(content="x" * 35_000, timestamp=0)])  # 10_000 tokens
    assert clamp_max_tokens_to_context(model, context, 40_000) == 50_000 - 10_000 - 4096


def test_clamp_keeps_requested_cap_when_context_is_roomy():
    model = small_model(context_window=100_000)
    context = TranscriptContext([UserMessage(content="hello", timestamp=0)])
    assert clamp_max_tokens_to_context(model, context, 8192) == 8192


def test_clamp_with_unknown_context_window_keeps_requested_cap():
    model = small_model(context_window=0)
    context = TranscriptContext([UserMessage(content="x" * 100_000, timestamp=0)])
    assert clamp_max_tokens_to_context(model, context, 8192) == 8192


def test_resolve_max_tokens_falls_back_to_model_cap():
    model = Model(id="m", name="m", api="anthropic-messages", provider="test",
                  base_url="https://example.test", context_window=200_000, max_tokens=8192)
    context = TranscriptContext([])
    assert resolve_max_tokens(model, context, None) == 8192
    assert resolve_max_tokens(model, context, SimpleStreamOptions()) == 8192


# ---------------------------------------------------------------------------
# Provider requests clamp before sending (acceptance: small window +
# large max_tokens + long input -> the request carries the clamped budget)
# ---------------------------------------------------------------------------

def long_context():
    return TranscriptContext([UserMessage(content="x" * 7000, timestamp=0)])


def build_cases():
    completions = Model(id="m", name="m", api="openai-completions", provider="test",
                        base_url="https://example.test", context_window=2000)
    responses = Model(id="m", name="m", api="openai-responses", provider="test",
                      base_url="https://example.test", context_window=2000)
    gemini = Model(id="m", name="m", api="google-generative-ai", provider="test",
                   base_url="https://example.test", context_window=2000)
    bedrock = Model(id="m", name="m", api="bedrock-converse-stream", provider="test",
                    base_url="https://bedrock", context_window=2000)
    claude = Model(id="m", name="m", api="anthropic-messages", provider="test",
                   base_url="https://example.test", context_window=2000)
    return [
        (anthropic, claude, lambda body: body["max_tokens"]),
        (openai_completions, completions, lambda body: body["max_tokens"]),
        (openai_responses, responses, lambda body: body["max_output_tokens"]),
        (google, gemini, lambda body: body["generationConfig"]["maxOutputTokens"]),
        (amazon_bedrock, bedrock, lambda body: body["inferenceConfig"]["maxTokens"]),
    ]


@pytest.mark.parametrize("module,model,field", build_cases(), ids=[
    "anthropic", "openai-completions", "openai-responses", "google", "bedrock",
])
def test_build_request_clamps_max_tokens_to_context(module, model, field):
    options = SimpleStreamOptions(max_tokens=8192)
    clamped = field(module.build_request(model, long_context(), options))
    assert clamped == 1  # 2000-token window, 2000 estimated input tokens: nothing left

    roomy_model = Model(id=model.id, name=model.name, api=model.api, provider=model.provider,
                        base_url=model.base_url, context_window=200_000, max_tokens=8192)
    assert field(module.build_request(roomy_model, long_context(), options)) == 8192


def test_anthropic_thinking_budget_shrinks_under_a_tight_window():
    model = small_model(reasoning=True)
    options = SimpleStreamOptions(reasoning="high", max_tokens=8192)
    body = anthropic.build_request(model, long_context(), options)
    # The clamped ceiling cannot host a valid budget (>= 1024), so thinking is
    # dropped instead of sending an invalid request.
    assert body["max_tokens"] == 1
    assert "thinking" not in body

    roomy = Model(id=model.id, name=model.name, api=model.api, provider=model.provider,
                  base_url=model.base_url, context_window=200_000, max_tokens=8192, reasoning=True)
    body = anthropic.build_request(roomy, long_context(), options)
    # Roomy window: the existing bump formula is untouched ("high" budget = 16384).
    assert body["max_tokens"] == 20480
    assert body["thinking"]["budget_tokens"] == 16384


def test_anthropic_thinking_shrinks_but_stays_valid_when_the_window_is_tight():
    # 10_000-token window with ~5_000 estimated input leaves 904 output tokens:
    # even after shrinking to the 1024 answer room the budget would fall below
    # Anthropic's 1024 minimum, so thinking is dropped.
    model = Model(id="m", name="m", api="anthropic-messages", provider="test",
                  base_url="https://example.test", context_window=10_000, max_tokens=8192, reasoning=True)
    body = anthropic.build_request(
        model, TranscriptContext([UserMessage(content="x" * 17_500, timestamp=0)]),
        SimpleStreamOptions(reasoning="high"),
    )
    assert body["max_tokens"] == 904
    assert "thinking" not in body

    # 20_000-token window: 10904 output tokens host a 9880 budget, which is
    # below the requested 16384 but still valid, so thinking stays enabled.
    model = Model(id="m", name="m", api="anthropic-messages", provider="test",
                  base_url="https://example.test", context_window=20_000, max_tokens=8192, reasoning=True)
    body = anthropic.build_request(
        model, TranscriptContext([UserMessage(content="x" * 17_500, timestamp=0)]),
        SimpleStreamOptions(reasoning="high"),
    )
    assert body["max_tokens"] == 10_904
    assert body["thinking"]["budget_tokens"] == 9_880


def test_bedrock_params_cannot_override_the_clamped_budget():
    model = small_model(api="bedrock-converse-stream")
    model.params = {"inferenceConfig": {"maxTokens": 4096}}
    options = SimpleStreamOptions(max_tokens=8192)
    body = amazon_bedrock.build_request(model, long_context(), options)
    assert body["inferenceConfig"]["maxTokens"] == 1

    # Without an explicit option, model params still supply the budget.
    body = amazon_bedrock.build_request(model, TranscriptContext([]), SimpleStreamOptions())
    assert body["inferenceConfig"]["maxTokens"] == 4096


def test_compaction_clamp_message_uses_the_shared_divisor():
    from pi.agent.compaction import clamp_message

    message = UserMessage(content="a" * 3000, timestamp=0)
    clamped = clamp_message(message, 1000)
    assert clamped is not message
    # Kept characters: max_tokens * 3.5 minus the 1024-character headroom
    # (previously max_tokens * 4), plus the truncation marker.
    max_chars = int(1000 * CHARS_PER_TOKEN) - 1024
    marker = "\n[... truncated {} characters ...]\n".format(len(message.content) - max_chars)
    assert len(clamped.content) == max_chars + len(marker)
    assert estimate_text_tokens(clamped.content) <= 1000
    # Tiny budgets keep the 256-character floor of kept text.
    tiny = clamp_message(message, 100)
    assert len(tiny.content) < len(message.content)
    assert estimate_text_tokens(tiny.content) <= 100
