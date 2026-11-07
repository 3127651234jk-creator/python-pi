"""Custom API model login: field validation, wizard flow, save transaction
and the three wired protocols (docs/specs/custom-api-login.md)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

import pi.coding_agent.custom_api as custom_api
import pi.coding_agent.config as config_module
from pi.ai.models import ModelError, resolve_model
from pi.ai.providers import STREAM_FNS
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import (DoneEvent, SystemMessage, TextContent, Tool, ToolCall,
                         TranscriptContext, UserMessage)
from pi.agent.agent import Agent
from pi.coding_agent.cli import async_main, custom_login_command
from pi.coding_agent.config import Configuration, read_json, write_json
from pi.coding_agent.custom_api import (API_FORMATS, CustomApiForm, CustomApiSaver, ConcurrentConfigChange,
                                        build_candidate_models, check_endpoint_suffix, endpoint_preview,
                                        run_custom_api_wizard, validate_api_key, validate_base_url,
                                        validate_display_name, validate_limits, validate_model_id,
                                        validate_positive_int, validate_provider_id)
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.session import SessionManager


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    for key in ("PI_PY_PROVIDER", "PI_PY_MODEL", "PI_PY_API_KEY", "PI_PY_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return Configuration(str(tmp_path))


class ScriptedIO:
    """WizardIO double: pops scripted answers in order and records prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.messages = []

    def _next(self, title):
        self.prompts.append(title)
        answer = self.answers.pop(0)
        return answer(self) if callable(answer) else answer

    async def choose(self, title, values, default=None):
        return self._next(title)

    async def text_input(self, title, default="", password=False, allow_cancel_word=True, validate=None):
        self.prompts.append((title, default, password))
        answer = self.answers.pop(0)
        answer = answer(self) if callable(answer) else answer
        if answer is None:
            return None
        if answer == ":cancel" and allow_cancel_word:
            return None
        return validate(answer) if validate is not None else answer

    async def confirm(self, title, default):
        return self._next(title)

    def say(self, message):
        self.messages.append(message)


def eof(_io):
    raise EOFError()


def happy_answers(api="openai-completions", provider_id="my-gateway", url="https://gateway.example.test/v1",
                  key="secret-key", model_id="vendor/model-a", review="save"):
    return [api, provider_id, url, key, model_id, "", False, review]


def form_after_wizard(config, registry, answers):
    io = ScriptedIO(answers)
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    return form, io


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_provider_id_rules(config):
    registry = ModelRegistry(config)
    validate_provider_id("my-gateway", registry.providers, config)
    for bad in ("", "Gateway", "9gate", "-gate", "a" * 65, "gate_way"):
        with pytest.raises(ValueError):
            validate_provider_id(bad, registry.providers, config)
    with pytest.raises(ValueError, match="reserved"):
        validate_provider_id("custom", registry.providers, config)
    write_json(config.agent_dir / "models.json", {"providers": {"taken": {"models": []}}})
    registry.reload()
    with pytest.raises(ValueError, match="already exists"):
        validate_provider_id("taken", registry.providers, config)
    write_json(config.agent_dir / "auth.json", {"orphan": {"type": "api_key", "key": "k"}})
    with pytest.raises(ValueError, match="/logout orphan"):
        validate_provider_id("orphan", registry.providers, config)


def test_base_url_rules():
    assert validate_base_url(" https://gateway.example.com/v1/ ") == "https://gateway.example.com/v1"
    assert validate_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"
    for bad in ("gateway.example.com", "ftp://gateway.example.com", "https:///v1",
                "https://user:pass@gateway.example.com", "https://gateway.example.com/v1?key=1",
                "https://gateway.example.com/v1#frag"):
        with pytest.raises(ValueError):
            validate_base_url(bad)


def test_key_model_and_limit_rules():
    assert validate_api_key("  sk-test ") == "sk-test"
    assert validate_api_key(":cancel") == ":cancel"  # legal key content
    with pytest.raises(ValueError):
        validate_api_key("   ")
    with pytest.raises(ValueError):
        validate_api_key("line1\nline2")
    assert validate_model_id(" vendor/model-a:7B ") == "vendor/model-a:7B"
    for bad in ("", "a\tb", "a\x00b"):
        with pytest.raises(ValueError):
            validate_model_id(bad)
    assert validate_display_name("", "vendor/model-a") == "vendor/model-a"
    assert validate_display_name("  My Model ", "x") == "My Model"
    assert validate_positive_int(" 4096 ", "Max") == 4096
    with pytest.raises(ValueError):
        validate_positive_int("0", "Max")
    with pytest.raises(ValueError):
        validate_positive_int("4k", "Max")
    with pytest.raises(ValueError):
        validate_limits(4096, 4096)
    with pytest.raises(ValueError):
        validate_limits(4096, 8192)
    validate_limits(32768, 4096)


def test_endpoint_preview_and_suffix_rules():
    assert endpoint_preview("openai-completions", "https://g.test/v1/") == "https://g.test/v1/chat/completions"
    assert endpoint_preview("anthropic-messages", "https://g.test") == "https://g.test/v1/messages"
    assert endpoint_preview("openai-responses", "https://g.test/v1") == "https://g.test/v1/responses"
    # Matching suffix suggests the root address.
    check = check_endpoint_suffix("https://g.test/v1/chat/completions", "openai-completions")
    assert check.suggested_root == "https://g.test/v1" and check.matching_api is None
    # Mismatched suffix points at the other protocol without switching silently.
    check = check_endpoint_suffix("https://g.test/v1/messages", "openai-completions")
    assert check.matching_api == "anthropic-messages"
    with pytest.raises(ValueError, match="/v1/v1/messages"):
        check_endpoint_suffix("https://g.test/v1", "anthropic-messages")


# ---------------------------------------------------------------------------
# Wizard flow
# ---------------------------------------------------------------------------


def test_wizard_happy_path_saves_configuration(config):
    registry = ModelRegistry(config)
    form, io = form_after_wizard(config, registry, happy_answers())
    assert form.provider_id == "my-gateway" and form.model_id == "vendor/model-a"
    assert form.display_name == "vendor/model-a" and form.api_key == "secret-key"
    assert form.context_window == 32768 and form.max_tokens == 4096
    assert "secret-key" not in "\n".join(io.messages)  # the key stays hidden on review
    assert any("filled (hidden)" in message for message in io.messages)
    models = read_json(config.agent_dir / "models.json")
    assert models["providers"]["my-gateway"] == {
        "api": "openai-completions", "base_url": "https://gateway.example.test/v1",
        "models": [{"id": "vendor/model-a", "name": "vendor/model-a", "context_window": 32768,
                    "max_tokens": 4096, "input": ["text"], "reasoning": False}],
    }
    assert "secret-key" not in json.dumps(models)
    auth = read_json(config.agent_dir / "auth.json")
    assert auth["my-gateway"] == {"type": "api_key", "key": "secret-key"}
    assert "my-gateway/vendor/model-a" in registry.models


def test_wizard_cancel_at_any_step_writes_nothing(config):
    registry = ModelRegistry(config)
    for partial in ([lambda io: None], ["openai-completions", lambda io: None],
                    ["openai-completions", "my-gateway", ":cancel"],
                    ["openai-completions", "my-gateway", "https://g.test/v1", eof],
                    happy_answers(review="cancel")):
        form, _io = form_after_wizard(config, registry, partial)
        assert form is None
        assert not (config.agent_dir / "models.json").exists()
        assert not (config.agent_dir / "auth.json").exists()


def test_wizard_back_preserves_values_and_keeps_key_in_memory(config):
    registry = ModelRegistry(config)
    answers = happy_answers()[:-1] + ["back"] + ["openai-completions", "other-gateway",
                                                 "https://g.test/v1", "", "vendor/model-a", "", False, "save"]
    io = ScriptedIO(answers)
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form is not None and form.provider_id == "other-gateway"
    assert form.api_key == "secret-key"  # kept from the first round, never re-typed
    titles = [prompt[0] for prompt in io.prompts if isinstance(prompt, tuple)]
    assert titles.count("Provider ID") == 2
    defaults = [prompt[1] for prompt in io.prompts if isinstance(prompt, tuple) and prompt[0] == "Provider ID"]
    assert defaults == ["", "my-gateway"]  # the second round pre-fills the kept value


def test_wizard_switches_format_on_pasted_endpoint(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(["anthropic-messages", "my-gateway", "https://g.test/v1/chat/completions", True, True,
                     "secret-key", "m", "", False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.api == "openai-completions"  # switched only after explicit confirmation
    assert form.base_url == "https://g.test/v1"
    labels = [prompt for prompt in io.prompts if isinstance(prompt, str)]
    assert any("Use that API format instead" in label for label in labels)
    assert any("Use the root https://g.test/v1 instead" in label for label in labels)


def test_wizard_declined_format_switch_stays_in_field(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(["anthropic-messages", "my-gateway", "https://g.test/v1/chat/completions", False,
                     "https://g.test/anthropic", "secret-key", "m", "", False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.api == "anthropic-messages" and form.base_url == "https://g.test/anthropic"


def test_wizard_requires_confirmation_for_plain_http(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(["openai-completions", "local", "http://localhost:8000/v1", "k", "m", "", False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.base_url == "http://localhost:8000/v1"
    assert not any("unencrypted" in str(prompt) for prompt in io.prompts)
    io = ScriptedIO(["openai-completions", "remote", "http://g.test/v1", False,
                     "https://g.test/v1", "k", "m", "", False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.base_url == "https://g.test/v1"
    assert any("unencrypted" in str(prompt) for prompt in io.prompts)


def test_wizard_reprompts_field_after_validation_failure(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(["openai-completions", "Bad ID", "my-gateway", "https://g.test/v1", "k", "m", "", False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.provider_id == "my-gateway"
    inputs = [prompt for prompt in io.prompts if isinstance(prompt, tuple) and prompt[0] == "Provider ID"]
    assert len(inputs) == 2
    assert any("error:" in message for message in io.messages)


def test_wizard_hands_field_rules_to_the_input_surface(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(happy_answers())
    seen = []
    original = io.text_input

    async def recording(title, default="", password=False, allow_cancel_word=True, validate=None):
        seen.append((title, validate is not None))
        return await original(title, default, password, allow_cancel_word, validate)

    io.text_input = recording
    assert asyncio.run(run_custom_api_wizard(config, registry, io)) is not None
    assert ("Provider ID", True) in seen and ("Model ID", True) in seen
    assert ("Base URL", False) in seen  # its own loop handles confirmations


def test_wizard_advanced_settings_flow(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(["openai-completions", "g", "https://g.test/v1", "k", "m", "", True,
                     "64000", "8192", True, True, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.context_window == 64000 and form.max_tokens == 8192
    assert form.image_input and form.reasoning
    # Invalid limits re-prompt the pair.
    io = ScriptedIO(["openai-completions", "g2", "https://g.test/v1", "k", "m", "", True,
                     "8192", "8192", "64000", "8192", False, False, "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form.context_window == 64000 and not form.reasoning


def test_unreadable_existing_files_block_the_wizard(config):
    registry = ModelRegistry(config)
    config.agent_dir.mkdir(parents=True, exist_ok=True)
    (config.agent_dir / "models.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read configuration"):
        asyncio.run(run_custom_api_wizard(config, registry, ScriptedIO(happy_answers())))
    assert "{not json" in (config.agent_dir / "models.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Save transaction
# ---------------------------------------------------------------------------


def test_save_preserves_existing_configuration(config):
    write_json(config.agent_dir / "models.json", {
        "providers": {"other": {"api": "openai-completions", "base_url": "https://o.test/v1",
                                "models": [{"id": "m", "name": "Other"}]}},
        "future_top_level_key": {"keep": True},
    })
    write_json(config.agent_dir / "auth.json", {"other": {"type": "api_key", "key": "o"}})
    registry = ModelRegistry(config)
    form, _io = form_after_wizard(config, registry, happy_answers())
    assert form is not None
    models = read_json(config.agent_dir / "models.json")
    assert models["future_top_level_key"] == {"keep": True}
    assert models["providers"]["other"] == {"api": "openai-completions", "base_url": "https://o.test/v1",
                                            "models": [{"id": "m", "name": "Other"}]}
    assert read_json(config.agent_dir / "auth.json")["other"]["key"] == "o"


def test_save_rejects_concurrent_change(config):
    registry = ModelRegistry(config)
    saver = CustomApiSaver(config)
    write_json(config.agent_dir / "models.json", {"providers": {"intruder": {"models": []}}})
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1", api_key="k", model_id="m")
    with pytest.raises(ConcurrentConfigChange):
        saver.save(registry, form)
    assert "intruder" in read_json(config.agent_dir / "models.json")["providers"]
    assert not (config.agent_dir / "auth.json").exists()


def test_wizard_reconfirms_after_concurrent_change(config):
    registry = ModelRegistry(config)
    io = ScriptedIO(happy_answers()[:-1] + [
        lambda io: (write_json(config.agent_dir / "models.json", {"providers": {"intruder": {"models": []}}}), "save")[1],
        "save"])
    form = asyncio.run(run_custom_api_wizard(config, registry, io))
    assert form is not None  # form kept, review re-shown, second Save succeeds
    assert "intruder" in read_json(config.agent_dir / "models.json")["providers"]
    assert "my-gateway" in read_json(config.agent_dir / "models.json")["providers"]
    assert any("changed while the wizard was open" in message for message in io.messages)


def test_reload_failure_rolls_back_both_files(config, monkeypatch):
    registry = ModelRegistry(config)
    monkeypatch.setattr(registry, "reload", lambda: (_ for _ in ()).throw(ModelError("boom")))
    saver = CustomApiSaver(config)
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1", api_key="k", model_id="m")
    with pytest.raises(ModelError, match="boom"):
        saver.save(registry, form)
    assert read_json(config.agent_dir / "models.json") == {}
    assert read_json(config.agent_dir / "auth.json") == {}
    # A previous configuration is restored verbatim.
    write_json(config.agent_dir / "models.json", {"providers": {"other": {"models": []}}})
    write_json(config.agent_dir / "auth.json", {"other": {"type": "api_key", "key": "o"}})
    saver = CustomApiSaver(config)
    with pytest.raises(ModelError):
        saver.save(registry, form)
    assert set(read_json(config.agent_dir / "models.json")["providers"]) == {"other"}
    assert read_json(config.agent_dir / "auth.json")["other"]["key"] == "o"


def test_models_write_failure_rolls_back_auth(config, monkeypatch):
    registry = ModelRegistry(config)
    saver = CustomApiSaver(config)
    real_write = custom_api.write_json
    attempts = {"models": 0}

    def flaky(path, value):
        if Path(path) == saver.models_path:
            attempts["models"] += 1
            if attempts["models"] == 1:
                raise OSError("disk full")
        return real_write(path, value)

    monkeypatch.setattr(custom_api, "write_json", flaky)
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1", api_key="k", model_id="m")
    with pytest.raises(OSError, match="disk full"):
        saver.save(registry, form)
    assert read_json(config.agent_dir / "auth.json") == {}
    assert not (config.agent_dir / "models.json").exists()


def test_rollback_failure_reports_repair_target(config, monkeypatch):
    registry = ModelRegistry(config)
    monkeypatch.setattr(registry, "reload", lambda: (_ for _ in ()).throw(ModelError("boom")))
    saver = CustomApiSaver(config)
    real_write = custom_api.write_json
    state = {"writes": 0}

    def flaky(path, value):
        if Path(path) in (saver.models_path, saver.auth_path):
            state["writes"] += 1
            if state["writes"] > 2:  # allow both save writes, fail every rollback write
                raise OSError("locked")
        return real_write(path, value)

    monkeypatch.setattr(custom_api, "write_json", flaky)
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1", api_key="k", model_id="m")
    with pytest.raises(custom_api.SaveError, match="rolled back"):
        saver.save(registry, form)
    # Both entries remain; the message named the files, never the key.
    assert "my-gateway" in read_json(config.agent_dir / "auth.json")
    assert "my-gateway" in read_json(config.agent_dir / "models.json")["providers"]
    assert "secret-key" not in str(custom_api.SaveError("probe"))


def test_candidate_validation_rejects_broken_existing_providers(config):
    form = CustomApiForm(provider_id="g", base_url="https://g.test/v1", api_key="k", model_id="m")
    broken = build_candidate_models({"providers": {"bad": {"api": "nonsense", "base_url": "https://x.test", "models": []}}}, form)
    with pytest.raises(ModelError):
        custom_api.validate_candidate(broken, form)


# ---------------------------------------------------------------------------
# Three protocols over mock HTTP
# ---------------------------------------------------------------------------


TEXT_SSE = {
    "openai-completions": 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":null}\n\ndata: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n',
    "anthropic-messages": 'event: message_start\ndata: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":1}}}\n\nevent: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\nevent: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
    "openai-responses": 'data: {"type":"response.output_text.delta","output_index":0,"delta":"Hello"}\n\ndata: {"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
}

TOOL_SSE = {
    "openai-completions": 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"x\\"}"}}]},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":null}\n\ndata: [DONE]\n\n',
    "anthropic-messages": 'event: message_start\ndata: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":1}}}\n\nevent: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"read"}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"x\\"}"}}\n\nevent: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
    "openai-responses": 'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call","name":"read","arguments":""}}\n\ndata: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\\"path\\":\\"x\\"}"}\n\ndata: {"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
}

EXPECTED = {
    "openai-completions": {"suffix": "/v1/chat/completions", "header": ("authorization", "Bearer")},
    "anthropic-messages": {"suffix": "/v1/messages", "header": ("x-api-key", "")},
    "openai-responses": {"suffix": "/v1/responses", "header": ("authorization", "Bearer")},
}


def mock_send(monkeypatch, sse):
    captured = {}

    async def fake_send(self, request, **kwargs):
        captured["request"] = request
        return httpx.Response(200, request=request, content=sse.encode())

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return captured


async def collect_stream(model, key, context):
    return [event async for event in STREAM_FNS[model.api](model, context, SimpleStreamOptions(api_key=key))]


@pytest.mark.parametrize("api", [name for name, _label in API_FORMATS])
def test_wizard_saved_model_streams_over_mock_http(config, monkeypatch, api):
    registry = ModelRegistry(config)
    # Claude Messages expects a root without /v1; the others sit under /v1.
    url = "https://g.test" if api == "anthropic-messages" else "https://g.test/v1"
    form, _io = form_after_wizard(config, registry, happy_answers(api=api, url=url))
    assert form is not None
    model = registry.resolve("my-gateway", "vendor/model-a")
    key = asyncio.run(model.get_api_key("my-gateway"))
    assert key == "secret-key"  # resolved from auth.json through the normal chain
    context = TranscriptContext([SystemMessage(content="s", timestamp=0, tools_added=[Tool("read", "Read", {"type": "object"})]),
                                 UserMessage(content="hello", timestamp=0)])
    captured = mock_send(monkeypatch, TEXT_SSE[api])
    events = asyncio.run(collect_stream(model, key, context))
    request = captured["request"]
    assert request.url.raw_path.decode().endswith(EXPECTED[api]["suffix"])
    header, prefix = EXPECTED[api]["header"]
    assert request.headers[header] == (prefix + " secret-key" if prefix else "secret-key")
    if api == "anthropic-messages":
        assert request.headers["anthropic-version"]
    assert json.loads(request.content)["model"] == "vendor/model-a"
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert "Hello" in "".join(block.text for block in done.message.content if isinstance(block, TextContent))
    # Tool calls stream through the same saved configuration.
    captured = mock_send(monkeypatch, TOOL_SSE[api])
    events = asyncio.run(collect_stream(model, key, context))
    done = next(event for event in events if isinstance(event, DoneEvent))
    call = next(block for block in done.message.content if isinstance(block, ToolCall))
    assert call.name == "read" and call.arguments == {"path": "x"}


def test_two_providers_share_model_id_independently(config, monkeypatch):
    monkeypatch.chdir(config.cwd)
    registry = ModelRegistry(config)
    CustomApiSaver(config).save(registry, CustomApiForm(
        provider_id="prov-a", api="openai-completions", base_url="https://a.test/v1",
        api_key="key-a", model_id="shared/model"))
    CustomApiSaver(config).save(registry, CustomApiForm(
        provider_id="prov-b", api="openai-responses", base_url="https://b.test/v1",
        api_key="key-b", model_id="shared/model"))
    model_a = registry.resolve("prov-a", "shared/model")
    model_b = registry.resolve("prov-b", "shared/model")
    assert model_a.base_url == "https://a.test/v1" and model_b.base_url == "https://b.test/v1"
    assert asyncio.run(model_a.get_api_key("prov-a")) == "key-a"
    assert asyncio.run(model_b.get_api_key("prov-b")) == "key-b"
    assert asyncio.run(async_main(["--logout", "prov-a"])) == 0
    registry.reload()
    # Logout clears only the credential; the model configuration stays usable
    # once the key is entered again.
    assert "prov-a/shared/model" in registry.models
    with pytest.raises(ModelError, match="No credentials for prov-a"):
        asyncio.run(model_a.get_api_key("prov-a"))
    model_b = registry.resolve("prov-b", "shared/model")
    assert asyncio.run(model_b.get_api_key("prov-b")) == "key-b"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_login_custom_cli_guards(config, monkeypatch, capsys):
    monkeypatch.chdir(config.cwd)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert asyncio.run(async_main(["--login", "custom", "--oauth"])) == 2
    assert "--oauth" in capsys.readouterr().err
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert asyncio.run(async_main(["--login", "custom"])) == 2
    assert "interactive terminal" in capsys.readouterr().err
    write_json(config.agent_dir / "models.json", {"providers": {"custom": {"models": []}}})
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert asyncio.run(async_main(["--login", "custom"])) == 2
    assert "reserved" in capsys.readouterr().err
    assert not (config.agent_dir / "auth.json").exists()


def test_login_custom_cli_text_wizard_saves_and_sets_default(config, monkeypatch, capsys):
    monkeypatch.chdir(config.cwd)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    inputs = iter(["", "my-gateway", "https://gateway.example.test/v1", "vendor/model-a", "", "", "", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "cli-secret")
    assert asyncio.run(async_main(["--login", "custom"])) == 0
    output = capsys.readouterr().out
    assert "saved provider 'my-gateway'" in output
    assert "cli-secret" not in output
    assert "connection has not been verified" in output
    settings = read_json(config.agent_dir / "settings.json")
    assert settings["provider"] == "my-gateway" and settings["model"] == "vendor/model-a"
    registry = ModelRegistry(config)
    assert "my-gateway/vendor/model-a" in registry.models


def test_save_many_updates_keys_in_one_write(config, monkeypatch):
    registry = ModelRegistry(config)
    writes = []
    real_write = config_module.write_json

    def counting(path, value):
        writes.append(Path(path))
        return real_write(path, value)

    monkeypatch.setattr(config_module, "write_json", counting)
    registry.config.save_many({"provider": "my-gateway", "model": "vendor/model-a"})
    assert writes == [config.agent_dir / "settings.json"]
    settings = read_json(config.agent_dir / "settings.json")
    assert settings["provider"] == "my-gateway" and settings["model"] == "vendor/model-a"


def test_custom_login_switches_model_and_forces_thinking_off(config, monkeypatch):
    registry = ModelRegistry(config)
    agent = Agent(resolve_model("faux"))
    agent.thinking_level = "high"
    session = SessionManager(cwd=str(config.cwd))
    io = ScriptedIO(happy_answers() + ["yes", "no"])
    monkeypatch.setattr("pi.coding_agent.cli.TextWizardIO", lambda: io)
    asyncio.run(custom_login_command(registry, agent=agent, session=session, ui=None))
    assert agent.model.provider == "my-gateway" and agent.model.id == "vendor/model-a"
    assert agent.thinking_level == "off"
    assert session.latest_custom("model_change")["provider"] == "my-gateway"
    assert session.latest_custom("thinking_level_change")["thinkingLevel"] == "off"
    assert "thinking" not in read_json(config.agent_dir / "settings.json")
    assert any("connection has not been verified" in message for message in io.messages)
    assert any(prompt == "Use this model now?" for prompt in io.prompts)


def test_custom_login_switch_failure_keeps_current_model(config, monkeypatch):
    registry = ModelRegistry(config)

    def boom(*args, **kwargs):
        raise ModelError("endpoint down")

    monkeypatch.setattr(registry, "resolve", boom)
    io = ScriptedIO(happy_answers() + ["yes", "no"])
    monkeypatch.setattr("pi.coding_agent.cli.TextWizardIO", lambda: io)
    agent = Agent(resolve_model("faux"))
    asyncio.run(custom_login_command(registry, agent=agent, session=None, ui=None))
    assert agent.model.provider == "faux"
    assert any("switching failed" in message for message in io.messages)


def test_custom_login_default_failure_reports_saved_model(config, monkeypatch):
    registry = ModelRegistry(config)

    def boom(*args, **kwargs):
        raise OSError("settings locked")

    monkeypatch.setattr(registry.config, "save_many", boom)
    io = ScriptedIO(happy_answers() + ["no", "yes"])
    monkeypatch.setattr("pi.coding_agent.cli.TextWizardIO", lambda: io)
    agent = Agent(resolve_model("faux"))
    asyncio.run(custom_login_command(registry, agent=agent, session=None, ui=None))
    assert any("setting the default failed" in message for message in io.messages)
    assert any("connection has not been verified" in message for message in io.messages)
    assert not (config.agent_dir / "settings.json").exists()
