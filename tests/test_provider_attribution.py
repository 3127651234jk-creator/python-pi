from pi.ai.models import Model
from pi.coding_agent.provider_attribution import (apply_attribution, default_attribution_headers,
                                                  is_install_telemetry_enabled,
                                                  merge_provider_attribution_headers)


def make_model(provider="openai", base_url="https://api.openai.com/v1", headers=None) -> Model:
    return Model(id="m", name="m", api="openai-completions", provider=provider,
                 base_url=base_url, headers=dict(headers or {}))


def test_telemetry_env_overrides_setting(monkeypatch):
    monkeypatch.setenv("PI_TELEMETRY", "0")
    assert is_install_telemetry_enabled({"enable_install_telemetry": True}) is False
    monkeypatch.setenv("PI_TELEMETRY", "1")
    assert is_install_telemetry_enabled({"enable_install_telemetry": False}) is True
    monkeypatch.delenv("PI_TELEMETRY")
    assert is_install_telemetry_enabled({}) is True
    assert is_install_telemetry_enabled({"enable_install_telemetry": False}) is False


def test_openrouter_attribution_headers():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    headers = default_attribution_headers(model, {})
    assert headers == {
        "HTTP-Referer": "https://pi.dev",
        "X-OpenRouter-Title": "pi",
        "X-OpenRouter-Categories": "cli-agent",
    }


def test_host_match_applies_without_provider_name():
    model = make_model(provider="custom", base_url="https://integrate.api.nvidia.com/v1")
    assert default_attribution_headers(model, {}) == {"X-BILLING-INVOKE-ORIGIN": "Pi"}
    cloudflare = make_model(provider="custom", base_url="https://gateway.ai.cloudflare.com/v1")
    assert default_attribution_headers(cloudflare, {}) == {"User-Agent": "pi-coding-agent"}


def test_unknown_provider_gets_no_headers():
    assert default_attribution_headers(make_model(), {}) is None


def test_merge_prefers_explicit_headers():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    merged = merge_provider_attribution_headers(model, {}, "sess", {"HTTP-Referer": "https://mine"})
    assert merged["HTTP-Referer"] == "https://mine"
    assert merged["X-OpenRouter-Title"] == "pi"


def test_opencode_session_headers():
    model = make_model(provider="opencode", base_url="https://opencode.ai/zen")
    merged = merge_provider_attribution_headers(model, {}, "abc")
    assert merged == {"x-opencode-session": "abc", "x-opencode-client": "pi"}
    other = make_model()
    assert merge_provider_attribution_headers(other, {}, "abc") is None


def test_apply_attribution_writes_model_headers():
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    added = apply_attribution(model, {})
    assert "X-OpenRouter-Title" in model.headers
    assert "X-OpenRouter-Title" in added
    # Applying again reports nothing new.
    assert apply_attribution(model, {}) == []
