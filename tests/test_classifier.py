"""Classifier API and codemode models.* tests (F-AI-03, F-AI-06, F-CA-04)."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.ai.models import Model, ModelCost, resolve_model
from pi.ai.providers import SimpleStreamOptions
from pi.ai.types import TranscriptContext
from pi.ai.providers import openai_decisions
from pi.ai.types import (
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierScoreQuestion,
    ImageContent,
    UserMessage,
)


def make_model() -> Model:
    return Model(id="gpt-6-luna", name="GPT-6 Luna", api="openai-decisions", provider="openai",
                 base_url="https://api.openai.com/v1", input=["text", "image"], context_window=922_000,
                 costs=ModelCost(input=0.1, output=0, cache_read=0, cache_write=0))


def context(**overrides) -> ClassifierContext:
    questions = {
        "mode": ClassifierChoiceQuestion(instructions="Pick a mode",
                                         criteria={"stay": "keep going", "stop": "finish"}),
        "quality": ClassifierScoreQuestion(instructions="Rate it", criteria=["low", "high"]),
        "safe": ClassifierBoolQuestion(instructions="Is it safe?", criteria={"true": "yes", "false": "no"}),
    }
    return ClassifierContext(state={"goal": "demo"}, questions=overrides.get("questions", questions),
                             images=overrides.get("images"))


def classify_with(monkeypatch, response, model=None, ctx=None, options=None):
    captured = {}

    async def fake_post(url, sent_model, body, sent_options):
        captured["url"] = url
        captured["body"] = body
        return response

    monkeypatch.setattr(openai_decisions, "_post_classifier_request", fake_post)
    result = asyncio.run(openai_decisions.classify_openai_decisions(
        model or make_model(), ctx or context(), options))
    return result, captured


def test_classify_maps_questions_and_parses_answers(monkeypatch):
    response = {
        "usage": {"input_tokens": 1000, "output_tokens": 10},
        "answers": [
            {"type": "choice", "name": "mode", "choice": "stay",
             "probabilities": [{"value": "stay", "probability": 0.9}, {"value": "stop", "probability": 0.1}],
             "confidence": 0.8},
            {"type": "score", "name": "quality", "score": 1, "confidence": 0.7},
            {"type": "predicate", "name": "safe", "probability": 0.95},
        ],
    }
    result, captured = classify_with(monkeypatch, response)
    assert captured["url"] == "https://api.openai.com/v1/decisions"
    assert captured["body"]["model"] == "gpt-6-luna"
    # bool maps to predicate with the meanings appended to the instructions.
    predicate = next(q for q in captured["body"]["questions"] if q["name"] == "safe")
    assert predicate["type"] == "predicate" and "True means: yes" in predicate["instructions"]
    assert result.stop_reason == "stop"
    assert result.answers["mode"].choice == "stay"
    assert result.answers["mode"].probabilities == {"stay": 0.9, "stop": 0.1}
    assert result.answers["quality"].score == 1
    assert result.answers["safe"].probability == 0.95
    # Input-only pricing from the catalog.
    assert result.usage.cost["total"] == pytest.approx(1000 / 1_000_000 * 0.1)


def test_classify_reports_refusal_and_missing_answers_as_errors(monkeypatch):
    result, _ = classify_with(monkeypatch, {"answers": [{"type": "refusal", "name": "mode"}]})
    assert result.stop_reason == "error" and "refused to answer mode" in result.error_message
    result, _ = classify_with(monkeypatch, {"answers": []})
    assert result.stop_reason == "error" and "did not return an answer for mode" in result.error_message


def test_classify_wire_input_with_images(monkeypatch):
    images = [ImageContent(data="aGVsbG8=", mime_type="image/png")]
    response = {"usage": {"input_tokens": 1, "output_tokens": 1}, "answers": [
        {"type": "predicate", "name": "safe", "probability": 0.5}]}
    result, captured = classify_with(
        monkeypatch, response, ctx=ClassifierContext(
            state={"x": 1},
            questions={"safe": ClassifierBoolQuestion(instructions="safe?", criteria={"true": "y", "false": "n"})},
            images=images))
    wire_input = captured["body"]["input"]
    assert wire_input[0]["content"][0] == {"type": "input_text", "text": '{"x": 1}'}
    assert wire_input[0]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
    assert result.answers["safe"].probability == 0.5


def test_classify_rejects_unknown_api_and_wrong_model(monkeypatch):
    async def fake_post(url, sent_model, body, sent_options):
        raise AssertionError("should not send")

    monkeypatch.setattr(openai_decisions, "_post_classifier_request", fake_post)
    wrong = Model(id="chat", name="c", api="openai-completions", provider="openai", base_url="x")
    result = asyncio.run(openai_decisions.classify_openai_decisions(wrong, context()))
    assert result.stop_reason == "error" and "Unsupported classifier API" in result.error_message


def test_system_one_maps_bool_to_noul(monkeypatch):
    """llama.cpp / OpenRouter's System One protocol: bool <-> noul (F-CA-05)."""
    from pi.ai.providers import typesafe_system_one
    from pi.ai.providers.typesafe_system_one import classify_system_one

    model = Model(id="julia-1", name="Julia", api="typesafe-system-one", provider="llama-cpp",
                  base_url="http://127.0.0.1:8080/v1", input=["text"])

    async def fake_post(url, sent_model, body, sent_options):
        assert url == "http://127.0.0.1:8080/v1/systemone"
        assert body["questions"]["safe"]["type"] == "noul"
        assert body["questions"]["mode"]["type"] == "choice"
        return {"usage": {"input_tokens": 30, "output_tokens": 2}, "answers": {
            "mode": {"type": "choice", "choice": "stay",
                     "probabilities": {"stay": 0.6, "stop": 0.4}, "confidence": 0.7},
            "quality": {"type": "score", "score": 1},
            "safe": {"type": "noul", "noul": 0.99},
        }}

    monkeypatch.setattr(typesafe_system_one, "_post_classifier_request", fake_post)
    result = asyncio.run(classify_system_one(model, context()))
    assert result.stop_reason == "stop"
    assert result.answers["safe"].type == "bool" and result.answers["safe"].probability == 0.99
    assert result.answers["mode"].choice == "stay"
    # Images are not supported on this protocol: an error result, not a request.
    imaged = context()
    imaged.images = [ImageContent(data="aGk=", mime_type="image/png")]
    result = asyncio.run(classify_system_one(model, imaged))
    assert result.stop_reason == "error" and "does not support image input" in result.error_message


def test_claude_haiku_5_5_adaptive_thinking():
    """Haiku 5.5: adaptive thinking + output_config.effort, no budget (F-AI-04)."""
    from pi.ai.providers import anthropic as anthropic_module

    catalog = resolve_model("anthropic", "claude-haiku-5-5", api_key="catalog-only")
    assert catalog.reasoning and catalog.context_window == 1_000_000
    # Long-prompt pricing tier (5x over 100k input tokens).
    assert catalog.costs.tiers and catalog.costs.tiers[0]["input_tokens_above"] == 100_000

    model = Model(id="claude-haiku-5-5", name="Haiku 5.5", api="anthropic-messages", provider="anthropic",
                  base_url="https://api.anthropic.com", reasoning=True, context_window=200_000,
                  max_tokens=64_000,
                  thinking_level_map={"off": None, "minimal": None, "low": "low", "medium": "medium",
                                      "high": "high", "xhigh": "xhigh", "max": "max"})
    body = anthropic_module.build_request(model, TranscriptContext([]), SimpleStreamOptions(reasoning="xhigh"))
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "xhigh"}
    assert "budget_tokens" not in str(body["thinking"])

    # minimal is mapped away: the level steps up to low.
    body = anthropic_module.build_request(model, TranscriptContext([]), SimpleStreamOptions(reasoning="minimal"))
    assert body["output_config"] == {"effort": "low"}
    # Requesting no thinking leaves thinking out entirely.
    body = anthropic_module.build_request(model, TranscriptContext([]), SimpleStreamOptions(reasoning=None))
    assert "thinking" not in body
    # Non-effort models keep the budget path.
    legacy = Model(id="claude-sonnet-4-5-20250929", name="Sonnet", api="anthropic-messages",
                   provider="anthropic", base_url="https://api.anthropic.com", reasoning=True,
                   context_window=200_000, max_tokens=64_000)
    body = anthropic_module.build_request(legacy, TranscriptContext([]), SimpleStreamOptions(reasoning="high"))
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 16384}


def test_bedrock_haiku_5_adaptive_thinking():
    """Bedrock haiku-5: adaptive thinking in additionalModelRequestFields (F-AI-04)."""
    from pi.ai.providers import amazon_bedrock

    model = Model(id="anthropic.claude-haiku-5-5", name="Haiku 5.5", api="bedrock-converse-stream",
                  provider="amazon-bedrock", base_url="https://bedrock", reasoning=True,
                  context_window=200_000, max_tokens=64_000,
                  thinking_level_map={"off": None, "minimal": None, "low": "low", "medium": "medium",
                                      "high": "high", "xhigh": "xhigh", "max": "max"})
    body = amazon_bedrock.build_request(model, TranscriptContext([]), SimpleStreamOptions(reasoning="xhigh"))
    fields = body["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "xhigh"}
    # Budget-based Claude models are untouched.
    legacy = Model(id="anthropic.claude-3-5-sonnet-20241022-v2:0", name="Sonnet", api="bedrock-converse-stream",
                   provider="amazon-bedrock", base_url="https://bedrock", reasoning=True,
                   context_window=200_000, max_tokens=64_000)
    body = amazon_bedrock.build_request(legacy, TranscriptContext([]), SimpleStreamOptions(reasoning="high"))
    assert body["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 16384}


def test_dispatch_wraps_errors_and_guards_image_inputs(monkeypatch):
    """Models.classify-style dispatch: failures are error results, not raises (F-AI-06)."""
    from pi.ai.providers import CLASSIFIER_FNS, classify

    text_only = Model(id="no-img", name="n", api="openai-decisions", provider="openai",
                      base_url="https://api.openai.com/v1", input=["text"])

    async def refuse(model, context, options=None):
        raise AssertionError("should not reach the adapter")

    monkeypatch.setitem(CLASSIFIER_FNS, "openai-decisions", refuse)

    ctx = ClassifierContext(state={}, questions={
        "safe": ClassifierBoolQuestion(instructions="s", criteria={"true": "y", "false": "n"})},
        images=[ImageContent(data="aGk=", mime_type="image/png")])
    result = asyncio.run(classify(text_only, ctx))
    assert result.stop_reason == "error" and "does not accept image inputs" in result.error_message


def test_codemode_models_classify_runs_in_host(tmp_path, monkeypatch):
    """A script's models.* calls resolve against the registry and classify in the host (F-CA-04)."""
    from pi.agent.types import AgentTool, AgentToolResult
    from pi.coding_agent.config import Configuration
    from pi.coding_agent.model_registry import ModelRegistry
    from pi.codemode.tool import create_codemode_tool
    from pi.ai.types import ClassifierBoolAnswer, ClassifierResult, TextContent

    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    registry = ModelRegistry(config)
    assert registry.get_model_of_type("classifier", "openai", "gpt-6-luna") is not None
    assert registry.get_model_of_type("classifier", "openai", "gpt-4o") is None

    async def fake_classify(model, context, options=None):
        assert model.id == "gpt-6-luna" and context.state == {"goal": "demo"}
        return ClassifierResult(api=model.api, provider=model.provider, model=model.id,
                                answers={"safe": ClassifierBoolAnswer(probability=0.9)},
                                timestamp=0)

    import pi.ai.providers as providers_module
    monkeypatch.setattr(providers_module, "classify", fake_classify)

    code = (
        "model = models.getModelOfType('classifier', 'openai', 'gpt-6-luna')\n"
        "available = models.getAvailableOfType('classifier', 'openai')\n"
        "assert any(m['id'] == 'gpt-6-luna' for m in available)\n"
        "result = models.classify(model, {'state': {'goal': 'demo'}, 'questions': {"
        "'safe': {'type': 'bool', 'instructions': 'safe?', 'criteria': {'true': 'y', 'false': 'n'}}}})\n"
        "result['answers']['safe']['probability']\n"
    )
    tool = create_codemode_tool([AgentTool(name="dummy", label="dummy", description="d",
                                           parameters={"type": "object", "properties": {}}, execute=None)],
                                model_registry=registry)
    result = asyncio.run(tool.execute("cm1", {"code": code}))
    assert not result.is_error
    assert "0.9" in "\n".join(block.text for block in result.content)
    classify_record = next(record for record in result.details["calls"] if record["name"] == "models.classify")
    assert classify_record["status"] == "ok"


def test_codemode_classify_rejects_bad_context(tmp_path):
    from pi.agent.types import AgentTool
    from pi.coding_agent.config import Configuration
    from pi.coding_agent.model_registry import ModelRegistry
    from pi.codemode.tool import create_codemode_tool

    registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path / "user")))
    code = (
        "model = models.getModelOfType('classifier', 'openai', 'gpt-6-luna')\n"
        "models.classify(model, {'state': 'not-an-object', 'questions': {}})\n"
    )
    tool = create_codemode_tool([AgentTool(name="dummy", label="dummy", description="d",
                                           parameters={"type": "object", "properties": {}}, execute=None)],
                                model_registry=registry)
    result = asyncio.run(tool.execute("cm2", {"code": code}))
    assert result.is_error
    assert "Expected context" in "\n".join(block.text for block in result.content)
