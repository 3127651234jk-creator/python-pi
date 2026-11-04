import asyncio
import base64
import json
import socket
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from pi.coding_agent.auth import AuthStore
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.oauth_callback import start_oauth_callback_server
from pi.coding_agent.subscription_auth import (login_claude, login_codex, login_copilot, login_kimi_coding, login_openai_chatgpt,
    login_radius, login_xai, normalize_radius_gateway, refresh_subscription, copilot_endpoint)
from pi.ai.types import TranscriptContext, UserMessage, ToolResultMessage, TextContent, SystemMessage, Tool
from pi.ai.providers import anthropic, openai_responses
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.providers.copilot import copilot_headers


def jwt(account="account"):
    payload = base64.urlsafe_b64encode(json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account}}).encode()).decode().rstrip("=")
    return "header." + payload + ".signature"


def test_claude_pkce_and_state_validation(monkeypatch):
    urls, requests = [], []
    async def post(self, url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(200, json={"access_token": "sk-ant-oat-test", "refresh_token": "refresh", "expires_in": 3600})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    async def prompt(message):
        state = parse_qs(urlsplit(urls[-1]).query)["state"][0]
        return "authorization#" + state
    record = asyncio.run(login_claude(urls.append, prompt))
    assert record["access"] == "sk-ant-oat-test"
    assert requests[0]["code_verifier"] == parse_qs(urlsplit(urls[0]).query)["state"][0]
    async def invalid(message): return "code#wrong-state"
    with pytest.raises(ValueError, match="state"):
        asyncio.run(login_claude(urls.append, invalid))
    assert len(requests) == 1


def test_codex_device_flow(monkeypatch):
    calls = []
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("usercode"):
            return httpx.Response(200, json={"device_auth_id": "device", "user_code": "ABCD", "interval": "1"})
        if url.endswith("deviceauth/token"):
            if len([u for u, _ in calls if u.endswith("deviceauth/token")]) == 1:
                return httpx.Response(403)
            return httpx.Response(200, json={"authorization_code": "authorization", "code_verifier": "verifier"})
        return httpx.Response(200, json={"access_token": jwt(), "refresh_token": "refresh", "expires_in": 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    record = asyncio.run(login_codex(lambda message: None))
    assert record["account_id"] == "account"
    assert calls[-1][1]["data"]["code_verifier"] == "verifier"


def test_copilot_device_flow_and_exchange(monkeypatch):
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        if url.endswith("device/code"):
            return httpx.Response(200, json={"device_code": "device", "user_code": "ABCD", "verification_uri": "https://github.com/login/device", "expires_in": 600})
        return httpx.Response(200, json={"access_token": "github-token"})
    async def get(self, url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer github-token"
        return httpx.Response(200, json={"token": "tid=x;proxy-ep=proxy.individual.githubcopilot.com;", "expires_at": time.time() + 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    record = asyncio.run(login_copilot(lambda message: None))
    assert record["refresh"] == "github-token"
    assert record["base_url"] == "https://api.individual.githubcopilot.com"
    with pytest.raises(ValueError):
        copilot_endpoint("proxy-ep=attacker.example;")


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex", "github-copilot"])
def test_subscription_refresh_and_failure_preserves_credentials(tmp_path, monkeypatch, provider):
    store = AuthStore(tmp_path / "auth.json")
    store.set(provider, {"type": "oauth", "access": "expired", "refresh": "refresh", "expires": 0})
    async def post(self, url, **kwargs):
        return httpx.Response(200, json={"access_token": jwt() if provider == "openai-codex" else "fresh", "refresh_token": "rotated", "expires_in": 3600})
    async def get(self, url, **kwargs):
        return httpx.Response(200, json={"token": "fresh", "expires_at": time.time() + 3600})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    assert asyncio.run(store.key(provider)) != "expired"
    record = store.get(provider)
    record["expires"] = 0
    store.set(provider, record)
    async def failed(self, url, **kwargs): return httpx.Response(401, json={"error": "private-value"})
    monkeypatch.setattr(httpx.AsyncClient, "post", failed)
    monkeypatch.setattr(httpx.AsyncClient, "get", failed)
    with pytest.raises(ValueError, match="401") as error:
        asyncio.run(store.key(provider))
    assert "private-value" not in str(error.value)
    assert store.get(provider) == record


def test_codex_endpoint_body_and_headers(tmp_path):
    registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path / "user")))
    registry.auth.set("openai-codex", {"type": "oauth", "access": jwt(), "refresh": "refresh", "expires": time.time() + 3600, "account_id": "account"})
    model = registry.resolve("openai-codex")
    asyncio.run(model.get_api_key(model.provider))
    assert model.headers["chatgpt-account-id"] == "account"
    body = openai_responses.build_request(model, TranscriptContext([UserMessage(content="hello", timestamp=0)]), SimpleStreamOptions())
    assert "max_output_tokens" not in body and not body["store"]
    assert body["instructions"] and model.base_url == "https://chatgpt.com/backend-api"


def test_claude_oauth_headers_and_tool_names(tmp_path, monkeypatch):
    registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path / "user")))
    registry.auth.set("anthropic", {"type": "oauth", "access": "sk-ant-oat-token", "refresh": "refresh", "expires": time.time() + 3600})
    model = registry.resolve("anthropic")
    captured = {}
    async def send(self, request, **kwargs):
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, request=request, content=b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async def check():
        key = await model.get_api_key(model.provider)
        context = TranscriptContext([SystemMessage(content="project", timestamp=0, tools_added=[Tool("read", "Read", {"type": "object"})]), UserMessage(content="hello", timestamp=0)])
        return [event async for event in anthropic.stream_anthropic(model, context, SimpleStreamOptions(api_key=key))]
    asyncio.run(check())
    assert captured["headers"]["authorization"] == "Bearer sk-ant-oat-token"
    assert "x-api-key" not in captured["headers"]
    assert captured["body"]["tools"][0]["name"] == "Read"
    assert "Claude Code" in captured["body"]["system"][0]["text"]


def test_copilot_initiator_changes_after_tool_result(tmp_path):
    registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path / "user")))
    model = registry.resolve("github-copilot")
    user = UserMessage(content="hello", timestamp=0)
    assert copilot_headers(model, TranscriptContext([user]))["X-Initiator"] == "user"
    result = ToolResultMessage(tool_call_id="call", tool_name="read", content=[TextContent(text="result")], is_error=False, timestamp=0)
    assert copilot_headers(model, TranscriptContext([user, result]))["X-Initiator"] == "agent"


DEVICE_ID = "0f0e0d0c-0b0a-4938-a726-001122334455"


def query_of(url):
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query).items() if values}


async def wait_for_output(outputs, count=1):
    while len(outputs) < count:
        await asyncio.sleep(0.01)
    return outputs


async def _async(value):
    return value


def chatgpt_token_response():
    return {"access_token": "chatgpt-access", "refresh_token": "rotated", "expires_in": 3600,
        "id_token": "id-token", "scope": "openid profile chatgpt.tokens.use.direct"}


def test_openai_chatgpt_browser_flow(monkeypatch):
    requests = []
    async def post(self, url, **kwargs):
        requests.append((url, kwargs))
        assert url == "https://auth.openai.com/api/accounts/oauth/token"
        return httpx.Response(200, json=chatgpt_token_response())
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    async def blocked(message):
        await asyncio.Event().wait()
    async def run():
        outputs = []
        task = asyncio.ensure_future(login_openai_chatgpt(outputs.append, blocked, DEVICE_ID))
        await asyncio.wait_for(wait_for_output(outputs), 5)
        params = query_of(outputs[0])
        assert params["client_id"] == "dynamic_agent_client" and params["agent_name_hint"] == "Pi"
        assert params["ext_agent_host_id"] == "urn:uuid:" + DEVICE_ID
        assert "chatgpt.tokens.use.direct" in params["scope"]
        assert params["code_challenge_method"] == "S256" and params["nonce"]
        async with httpx.AsyncClient(trust_env=False) as browser:
            page = await browser.get("http://127.0.0.1:1455/auth/callback?" + urlencode(
                {"code": "authorization", "state": params["state"], "client_id": "issued"}))
        record = await asyncio.wait_for(task, 5)
        return outputs, page, record, requests
    outputs, page, record, requests = asyncio.run(run())
    assert page.status_code == 200 and "Signed in to OpenAI" in page.text
    assert record["access"] == "chatgpt-access" and record["client_id"] == "issued"
    assert record["expires"] < time.time() + 3600 - 170
    assert requests[0][1]["data"]["client_id"] == "issued" and "code_verifier" in requests[0][1]["data"]


def test_openai_chatgpt_manual_flow_state_and_redirect_checks(monkeypatch):
    async def post(self, url, **kwargs):
        return httpx.Response(200, json=chatgpt_token_response())
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    async def run(paste):
        outputs = []
        async def prompt(message):
            return paste(query_of(outputs[0])["state"])
        return await asyncio.wait_for(login_openai_chatgpt(outputs.append, prompt, DEVICE_ID), 5)
    record = asyncio.run(run(lambda s: "http://127.0.0.1:1455/auth/callback?" + urlencode(
        {"code": "authorization", "state": s, "client_id": "issued"})))
    assert record["client_id"] == "issued"
    with pytest.raises(ValueError, match="must start with"):
        asyncio.run(run(lambda s: "http://attacker.example/callback?code=x&state=" + s + "&client_id=issued"))
    with pytest.raises(ValueError, match="state"):
        asyncio.run(run(lambda s: "http://127.0.0.1:1455/auth/callback?code=x&state=wrong&client_id=issued"))


def test_openai_chatgpt_rejects_bad_device_id_and_busy_port(monkeypatch):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 1455))
    blocker.listen(1)
    try:
        async def run():
            async def prompt(message):
                await asyncio.Event().wait()
            return await asyncio.wait_for(login_openai_chatgpt(lambda message: None, prompt, DEVICE_ID), 5)
        with pytest.raises(ValueError, match="Port 1455 is in use"):
            asyncio.run(run())
        async def prompt(message):
            await asyncio.Event().wait()
        with pytest.raises(ValueError, match="device ID"):
            asyncio.run(login_openai_chatgpt(lambda message: None, prompt, "not-a-uuid"))
    finally:
        blocker.close()


def test_chatgpt_refresh_requires_and_keeps_client_id(monkeypatch):
    async def post(self, url, **kwargs):
        assert kwargs["data"]["grant_type"] == "refresh_token" and kwargs["data"]["client_id"] == "issued"
        return httpx.Response(200, json=chatgpt_token_response())
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    record = asyncio.run(refresh_subscription("openai", {"type": "oauth", "access": "old", "refresh": "refresh", "expires": 0, "client_id": "issued"}))
    assert record["client_id"] == "issued" and record["access"] == "chatgpt-access"
    with pytest.raises(ValueError, match="client ID"):
        asyncio.run(refresh_subscription("openai", {"type": "oauth", "access": "old", "refresh": "refresh", "expires": 0}))


def test_kimi_device_flow(monkeypatch):
    polls = []
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        if url.endswith("/api/oauth/device_authorization"):
            assert kwargs["data"] == {"client_id": "17e5f671-d194-4dfb-9706-5516cb48c098"}
            return httpx.Response(200, json={"device_code": "device", "user_code": "ABCD",
                "verification_uri": "https://auth.kimi.com/device", "verification_uri_complete": "https://auth.kimi.com/device?code=ABCD",
                "interval": 1, "expires_in": 600})
        polls.append(kwargs["data"])
        if len(polls) == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "kimi-access", "refresh_token": "kimi-refresh", "expires_in": 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    record = asyncio.run(login_kimi_coding(lambda message: None))
    assert record["access"] == "kimi-access" and record["expires"] > time.time() + 3500
    assert polls[0]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"


def test_kimi_refresh_retries_and_unauthorized(monkeypatch):
    attempts = []
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        attempts.append(kwargs["data"]["refresh_token"])
        if len(attempts) <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    credential = {"type": "oauth", "access": "old", "refresh": "refresh", "expires": 0}
    record = asyncio.run(refresh_subscription("kimi-coding", credential))
    assert record["access"] == "fresh" and len(attempts) == 3
    async def denied(self, url, **kwargs):
        return httpx.Response(401, json={"error": "invalid_grant", "error_description": "revoked"})
    monkeypatch.setattr(httpx.AsyncClient, "post", denied)
    with pytest.raises(ValueError, match="unauthorized.*revoked"):
        asyncio.run(refresh_subscription("kimi-coding", credential))


def test_xai_device_flow_and_refresh_keeps_previous_refresh_token(monkeypatch):
    polls = []
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        if url.endswith("/oauth2/device/code"):
            assert kwargs["data"]["referrer"] == "pi" and kwargs["data"]["scope"].startswith("openid")
            return httpx.Response(200, json={"device_code": "device", "user_code": "ABCD",
                "verification_uri": "https://auth.x.ai/device", "expires_in": 600})
        polls.append(kwargs["data"])
        if len(polls) == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        if len(polls) == 2:
            return httpx.Response(400, json={"error": "slow_down", "interval": 3})
        return httpx.Response(200, json={"access_token": "grok-access", "refresh_token": "grok-refresh", "expires_in": 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    record = asyncio.run(login_xai(lambda message: None))
    assert record["access"] == "grok-access" and record["expires"] < time.time() + 3600 - 290
    async def refresh(self, url, **kwargs):
        return httpx.Response(200, json={"access_token": "grok-fresh"})
    monkeypatch.setattr(httpx.AsyncClient, "post", refresh)
    record = asyncio.run(refresh_subscription("xai", record))
    assert record["refresh"] == "grok-refresh"
    async def insecure(self, url, **kwargs):
        return httpx.Response(200, json={"device_code": "d", "user_code": "A",
            "verification_uri": "http://auth.x.ai/device", "expires_in": 600})
    monkeypatch.setattr(httpx.AsyncClient, "post", insecure)
    with pytest.raises(ValueError, match="Invalid xAI OAuth response field"):
        asyncio.run(login_xai(lambda message: None))


def test_radius_browser_flow(monkeypatch):
    exchanges = []
    original_get = httpx.AsyncClient.get
    async def get(self, url, **kwargs):
        if url != "https://gw.example/v1/oauth":
            return await original_get(self, url, **kwargs)
        return httpx.Response(200, json={"authorizationEndpoint": "https://gw.example/oauth/authorize"})
    async def post(self, url, **kwargs):
        exchanges.append((url, kwargs["data"]))
        return httpx.Response(200, json={"access_token": "radius-access", "refresh_token": "radius-refresh",
            "expires_in": 3600, "scope": "gateway"})
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    async def run():
        outputs = []
        async def choose(message):
            return ""
        task = asyncio.ensure_future(login_radius(outputs.append, choose, "gw.example"))
        await asyncio.wait_for(wait_for_output(outputs), 5)
        assert outputs[0] == "Listening for OAuth callback on http://127.0.0.1:1456/oauth/callback"
        params = query_of(outputs[1])
        assert params["client_id"] == "pi-gateway" and params["handoff"] == "url" and params["code_challenge_method"] == "S256"
        async with httpx.AsyncClient(trust_env=False) as browser:
            page = await browser.get("http://127.0.0.1:1456/oauth/callback?" + urlencode(
                {"code": "authorization", "state": params["state"]}))
        record = await asyncio.wait_for(task, 5)
        return page, record, exchanges
    page, record, exchanges = asyncio.run(run())
    assert page.status_code == 200 and "Signed in to Radius" in page.text
    assert record["access"] == "radius-access" and record["gateway"] == "https://gw.example" and record["scope"] == "gateway"
    assert exchanges[0][0] == "https://gw.example/v1/oauth/token" and exchanges[0][1]["grant_type"] == "authorization_code"


def test_radius_device_flow_and_refresh(monkeypatch):
    polls = []
    async def sleep(seconds): pass
    async def post(self, url, **kwargs):
        if url.endswith("/v1/oauth/device"):
            return httpx.Response(200, json={"device_code": "device", "user_code": "ABCD",
                "verification_uri": "https://gw.example/link", "expires_in": 600, "interval": 1})
        polls.append(kwargs["data"])
        if len(polls) == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "radius-access", "refresh_token": "radius-refresh", "expires_in": 3600})
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    record = asyncio.run(login_radius(lambda message: None, lambda message: _async("device-code"), "https://gw.example"))
    assert record["gateway"] == "https://gw.example"
    assert polls[0]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    async def refresh(self, url, **kwargs):
        assert url == "https://gw.example/v1/oauth/token" and kwargs["data"]["grant_type"] == "refresh_token"
        return httpx.Response(200, json={"access_token": "radius-fresh", "refresh_token": "rotated", "expires_in": 3600})
    monkeypatch.setattr(httpx.AsyncClient, "post", refresh)
    assert asyncio.run(refresh_subscription("radius", record))["access"] == "radius-fresh"
    with pytest.raises(ValueError, match="gateway"):
        asyncio.run(refresh_subscription("radius", {"type": "oauth", "access": "old", "refresh": "refresh", "expires": 0}))


@pytest.mark.parametrize("provider,extra", [
    ("openai", {"client_id": "issued"}),
    ("kimi-coding", {}),
    ("xai", {}),
    ("radius", {"gateway": "https://gw.example"}),
])
def test_new_subscription_refresh_preserves_credentials_on_failure(tmp_path, monkeypatch, provider, extra):
    store = AuthStore(tmp_path / "auth.json")
    record = {"type": "oauth", "access": "expired", "refresh": "refresh", "expires": 0}
    record.update(extra)
    store.set(provider, record)
    async def post(self, url, **kwargs):
        return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600,
            "scope": "openid chatgpt.tokens.use.direct"})
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    assert asyncio.run(store.key(provider)) == "fresh"
    stored = store.get(provider)
    stored["expires"] = 0
    store.set(provider, stored)
    async def failed(self, url, **kwargs):
        return httpx.Response(401, json={"error": "denied"})
    monkeypatch.setattr(httpx.AsyncClient, "post", failed)
    with pytest.raises(ValueError):
        asyncio.run(store.key(provider))
    assert store.get(provider)["access"] == "fresh"


def test_normalize_radius_gateway():
    assert normalize_radius_gateway("gw.example") == "https://gw.example"
    assert normalize_radius_gateway("http://gw.example/") == "http://gw.example"
    assert normalize_radius_gateway(None) == "https://radius.pi.dev"


def test_oauth_callback_server_semantics():
    async def run():
        async def ok(code):
            return "exchanged-" + code
        server = await start_oauth_callback_server("Test", "127.0.0.1", 0, "/cb", state="s1", complete=ok)
        try:
            base = server.redirect_uri
            assert base.endswith("/cb")
            # trust_env=False: a system proxy would otherwise answer loopback GETs with 502.
            async with httpx.AsyncClient(trust_env=False) as browser:
                page = await browser.get(base + "?code=a&state=nope")
                assert page.status_code == 400 and "State mismatch" in page.text
                page = await browser.get(base + "/other")
                assert page.status_code == 404
                page = await browser.get(base + "?error=denied&state=s1")
                assert page.status_code == 400 and "Test authorization failed" in page.text
                with pytest.raises(ValueError, match="denied"):
                    await asyncio.wait_for(server.wait(), 2)
                page = await browser.get(base + "?code=a&state=s1")
                assert page.status_code == 409
        finally:
            await server.close()
        second = await start_oauth_callback_server("Test", "127.0.0.1", 0, "/cb", state="s1", complete=ok)
        try:
            async with httpx.AsyncClient(trust_env=False) as browser:
                page = await browser.get(second.redirect_uri + "?code=b&state=s1")
                assert page.status_code == 200 and "Signed in to Test" in page.text
                page = await browser.get(second.redirect_uri + "?code=c&state=s1")
                assert page.status_code == 409
            assert await asyncio.wait_for(second.wait(), 2) == "exchanged-b"
        finally:
            await second.close()
        third = await start_oauth_callback_server("Test", "127.0.0.1", 0, "/cb", timeout_ms=50)
        try:
            with pytest.raises(ValueError, match="timed out"):
                await asyncio.wait_for(third.wait(), 2)
        finally:
            await third.close()
        fourth = await start_oauth_callback_server("Test", "127.0.0.1", 0, "/cb")
        fourth.cancel()
        assert await asyncio.wait_for(fourth.wait(), 2) is None
        await fourth.close()
    asyncio.run(run())
