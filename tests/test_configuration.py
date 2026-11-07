import asyncio
import json
import os
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from pi.agent.agent import Agent
from pi.ai.models import ModelError, resolve_model
from pi.ai.providers import STREAM_FNS
from pi.ai.providers.faux import reset_script_index
from pi.coding_agent.auth import AuthStore, login_device, login_openrouter
from pi.coding_agent.cli import (CliRenderer, async_main, authenticate, build_parser, interactive_loop,
                                 login_provider_options, make_agent)
from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.model_registry import ModelRegistry


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    for key in ("PI_PY_PROVIDER", "PI_PY_MODEL", "PI_PY_API_KEY", "PI_PY_BASE_URL", "PI_PY_THINKING", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return Configuration(str(tmp_path))


def test_settings_precedence_and_validation(config, monkeypatch):
    config.save("provider", "faux")
    config.save("thinking", "low")
    config.save("thinking", "high", project=True)
    registry = ModelRegistry(config)
    agent = make_agent(build_parser().parse_args([]), str(config.cwd), registry)
    assert agent.model.provider == "faux"
    assert agent.thinking_level == "high"
    monkeypatch.setenv("PI_PY_THINKING", "medium")
    assert make_agent(build_parser().parse_args([]), str(config.cwd), registry).thinking_level == "medium"
    assert make_agent(build_parser().parse_args(["--thinking", "off"]), str(config.cwd), registry).thinking_level == "off"
    with pytest.raises(ValueError):
        config.save("compaction", {"summary_tokens": -1})
    assert "compaction" not in config.settings


def test_model_capabilities_and_credentials(config, monkeypatch):
    monkeypatch.setenv("CUSTOM_KEY", "environment")
    write_json(config.agent_dir / "models.json", {"providers": {"local": {
        "api": "openai-completions", "base_url": "http://localhost:8000/v1",
        "api_key": "${CUSTOM_KEY}", "headers": {"X-Custom": "$CUSTOM_KEY"},
        "models": [{"id": "coder", "reasoning": True, "input": ["text", "image"],
                    "context_window": 32000, "max_tokens": 4096, "costs": {"input": 1.5}}],
    }}})
    registry = ModelRegistry(config)
    model = registry.resolve("local", "coder")
    assert model.context_window == 32000 and model.max_tokens == 4096
    assert model.reasoning and model.input == ["text", "image"]
    assert model.costs.input == 1.5 and model.headers["X-Custom"] == "environment"
    assert asyncio.run(model.get_api_key("local")) == "environment"
    registry.auth.set("local", {"type": "api_key", "key": "stored"})
    assert asyncio.run(model.get_api_key("local")) == "stored"
    override = registry.resolve("local", "coder", api_key="explicit")
    assert asyncio.run(override.get_api_key("local")) == "explicit"
    registry.auth.logout("local")
    assert registry.auth.get("local") is None
    if os.name != "nt":
        assert (registry.auth.path.stat().st_mode & 0o777) == 0o600


def test_no_auth_local_provider_and_model_switch_metadata(config):
    write_json(config.agent_dir / "models.json", {"providers": {"local": {
        "base_url": "http://localhost:8000/v1", "auth_required": False, "models": [{"id": "coder"}],
    }}})
    registry = ModelRegistry(config)
    assert asyncio.run(registry.resolve("local", "coder").get_api_key("local")) == ""
    assert registry.resolve("openai", "gpt-4o").max_tokens == 16384
    with pytest.raises(ModelError):
        asyncio.run(registry.resolve("openai", "gpt-4o").get_api_key("openai"))


@pytest.mark.parametrize("entry", [{"max_tokens": 0}, {"context_window": "bad"}, {"input": ["audio"]}, {"reasoning": "yes"}, {"unexpected": True}])
def test_invalid_model_settings(config, entry):
    write_json(config.agent_dir / "models.json", {"providers": {"openai": {"models": [{"id": "gpt-4o", **entry}]}}})
    with pytest.raises(ModelError):
        ModelRegistry(config)


def test_refresh_is_persisted_and_used(config, monkeypatch):
    store = AuthStore(config.agent_dir / "auth.json")
    store.set("custom", {"type": "oauth", "access": "expired", "refresh": "old-refresh", "expires": 0})
    calls = []

    async def post(self, url, **kwargs):
        calls.append(kwargs["data"])
        return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    oauth = {"token_url": "https://example.test/token", "client_id": "client"}
    assert asyncio.run(store.key("custom", oauth)) == "new-access"
    assert store.get("custom")["refresh"] == "new-refresh"
    assert asyncio.run(store.key("custom", oauth)) == "new-access"
    assert len(calls) == 1 and calls[0]["refresh_token"] == "old-refresh"


def test_failed_refresh_preserves_credentials(config, monkeypatch):
    store = AuthStore(config.agent_dir / "auth.json")
    credential = {"type": "oauth", "access": "expired", "refresh": "secret", "expires": 0}
    store.set("custom", credential)

    async def post(self, url, **kwargs):
        return httpx.Response(401, json={"error": "secret must not appear"})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with pytest.raises(ValueError, match="HTTP 401") as error:
        asyncio.run(store.key("custom", {"token_url": "https://example.test", "client_id": "client"}))
    assert "secret" not in str(error.value)
    assert store.get("custom") == credential


def test_openrouter_pkce_loopback(monkeypatch):
    async def check():
        captured = {}
        tasks = []

        async def post(self, url, **kwargs):
            captured["exchange"] = kwargs["json"]
            return httpx.Response(200, json={"key": "oauth-key"})

        async def visit(url):
            params = parse_qs(urlsplit(url).query)
            callback = urlsplit(params["callback_url"][0])
            captured["params"] = params
            reader, writer = await asyncio.open_connection(callback.hostname, callback.port)
            writer.write(f"GET {callback.path}?code=test-code HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
            await writer.drain()
            assert b"200 OK" in await reader.read()
            writer.close()
            await writer.wait_closed()

        def notify(url):
            tasks.append(asyncio.create_task(visit(url)))

        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        credential = await login_openrouter(notify, timeout=2)
        await asyncio.gather(*tasks)
        assert credential["access"] == "oauth-key"
        assert captured["exchange"]["code"] == "test-code"
        assert captured["params"]["code_challenge_method"] == ["S256"]
        assert captured["exchange"]["code_verifier"] != captured["params"]["code_challenge"][0]
    asyncio.run(check())


def test_device_login_pending_and_slowdown(monkeypatch):
    replies = iter([
        {"verification_uri": "https://example.test/activate", "user_code": "ABCD", "device_code": "device", "expires_in": 60, "interval": 1},
        {"error": "authorization_pending"}, {"error": "slow_down"},
        {"access_token": "device-access", "refresh_token": "refresh", "expires_in": 3600},
    ])
    sleeps = []

    async def post(self, url, **kwargs):
        data = next(replies)
        return httpx.Response(400 if "error" in data else 200, json=data)

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    messages = []
    credential = asyncio.run(login_device({"device_url": "https://example.test/device", "token_url": "https://example.test/token", "client_id": "client"}, messages.append))
    assert credential["access"] == "device-access"
    assert sleeps == [1, 1, 6]
    assert "ABCD" in messages[0]


def test_cli_list_models_needs_no_credentials(config, monkeypatch, capsys):
    monkeypatch.chdir(config.cwd)
    assert asyncio.run(async_main(["--list-models"])) == 0
    assert "openrouter/openrouter/auto" in capsys.readouterr().out
    assert not (config.cwd / ".pi" / "sessions").exists()


def test_agent_resolves_credentials_on_each_request(config, monkeypatch):
    registry = ModelRegistry(config)
    registry.auth.set("openai", {"type": "api_key", "key": "first"})
    agent = make_agent(build_parser().parse_args(["--provider", "openai"]), str(config.cwd), registry)
    captured = []

    async def stream(model, context, options):
        captured.append(options.api_key)
        async for event in STREAM_FNS["faux"](model, context, options):
            yield event

    monkeypatch.setitem(STREAM_FNS, "openai-completions", stream)

    async def check():
        reset_script_index()
        await agent.prompt("one")
        registry.auth.set("openai", {"type": "api_key", "key": "second"})
        reset_script_index()
        await agent.prompt("two")
    asyncio.run(check())
    assert captured == ["first", "second"]


def test_api_key_login_logout(config, monkeypatch, capsys):
    registry = ModelRegistry(config)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "private-key")
    asyncio.run(authenticate(registry, "openai"))
    assert registry.auth.get("openai")["key"] == "private-key"
    assert "private-key" not in capsys.readouterr().out
    assert "private-key" not in repr(registry.resolve("openai", api_key="private-key"))
    assert asyncio.run(async_main(["--logout", "openai"])) == 0
    assert registry.auth.get("openai") is None


class PickerUI:
    def __init__(self, inputs, selections):
        self.inputs = iter(inputs)
        self.selections = list(selections)
        self.prompts = []

    async def read(self, running=False):
        return next(self.inputs)

    async def choose(self, title, values):
        self.prompts.append((title, [label for _value, label in values]))
        return self.selections.pop(0)


def test_login_without_provider_uses_pickers(config, monkeypatch):
    registry = ModelRegistry(config)
    recorded = []

    async def authenticate(reg, provider, oauth=False):
        recorded.append((provider, oauth))

    monkeypatch.setattr("pi.coding_agent.cli.authenticate", authenticate)
    ui = PickerUI(["/login", "/exit"], ["oauth", "radius"])
    asyncio.run(interactive_loop(Agent(resolve_model("faux")), CliRenderer(quiet=True), None, registry, ui=ui))
    assert recorded == [("radius", True)]
    assert ui.prompts[0][0] == "Sign in"
    assert "Add custom API model" in ui.prompts[0][1]
    assert ui.prompts[1][0] == "Provider"
    assert "radius" in ui.prompts[1][1] and "anthropic" in ui.prompts[1][1]


def test_login_without_provider_lists_providers_in_plain_mode(config, monkeypatch, capsys):
    registry = ModelRegistry(config)
    inputs = iter(["/login", "1", ":cancel", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    asyncio.run(interactive_loop(Agent(resolve_model("faux")), CliRenderer(quiet=True), None, registry))
    output = capsys.readouterr().out
    assert "Sign in" in output and "Add custom API model" in output
    assert "Provider" in output
    assert "radius" in output and "anthropic" in output


def test_login_provider_options_cover_both_methods(config):
    registry = ModelRegistry(config)
    oauth = {name for name, _label in login_provider_options(registry, True)}
    api_key = {name for name, _label in login_provider_options(registry, False)}
    assert {"anthropic", "openai", "openrouter", "radius", "openai-codex"} <= oauth
    assert {"anthropic", "openai"} <= api_key
    assert "faux" not in api_key
    assert "radius" not in api_key


def test_openrouter_timeout_closes_listener():
    async def check():
        urls = []
        with pytest.raises(asyncio.TimeoutError):
            await login_openrouter(urls.append, timeout=0.01)
        callback = urlsplit(parse_qs(urlsplit(urls[0]).query)["callback_url"][0])
        with pytest.raises(OSError):
            await asyncio.open_connection(callback.hostname, callback.port)
    asyncio.run(check())


def test_failed_catalog_reload_keeps_previous_catalog(config):
    registry = ModelRegistry(config)
    write_json(config.agent_dir / "models.json", {"providers": {"openai": {"api": "unsupported"}}})
    with pytest.raises(ModelError):
        registry.reload()
    assert registry.resolve("openai", "gpt-4o").max_tokens == 16384
