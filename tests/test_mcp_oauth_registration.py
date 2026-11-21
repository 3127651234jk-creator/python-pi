"""Dynamic client registration parity (O04, O05)."""

import asyncio
import json

import httpx
import pytest

from pi.coding_agent.mcp_oauth import (
    McpOAuthError,
    OAuthRegistrationError,
    application_type,
    callback_settings,
    client_metadata,
    register_client,
)


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("redirect_uris,expected", [
    (["http://127.0.0.1:1234/callback"], "native"),
    (["http://localhost:1234/callback"], "native"),
    (["http://[::1]:1234/callback"], "native"),
    (["com.example.app:/callback"], "native"),
    (["https://example.test/callback"], "web"),
    (["not a url", "https://example.test/callback"], "web"),
    ([], "web"),
])
def test_application_type_matches_ts_fixtures(redirect_uris, expected):
    assert application_type(redirect_uris) == expected


def test_registration_sends_derived_application_type_and_scope():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"client_id": "issued", "redirect_uris": [
            "http://127.0.0.1:4567/callback"]}, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            metadata = await register_client(
                client, "https://auth.test",
                client_metadata({}, "http://127.0.0.1:4567/callback"),
                {"registration_endpoint": "https://auth.test/register"}, "tools.read")
        body = json.loads(requests[0].content)
        assert requests[0].method == "POST"
        assert body["application_type"] == "native"
        assert body["redirect_uris"] == ["http://127.0.0.1:4567/callback"]
        assert body["grant_types"] == ["authorization_code", "refresh_token"]
        assert body["response_types"] == ["code"]
        assert body["token_endpoint_auth_method"] == "none"
        assert body["client_name"] == "pi"
        assert body["scope"] == "tools.read"
        assert metadata["client_id"] == "issued"

    run(scenario())


def test_explicit_application_type_overrides_derivation():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"client_id": "issued"}, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            metadata = client_metadata({"application_type": "web"}, "http://127.0.0.1:4567/callback")
            await register_client(client, "https://auth.test", metadata,
                                  {"registration_endpoint": "https://auth.test/register"})
        assert json.loads(requests[0].content)["application_type"] == "web"

    run(scenario())


def test_missing_registration_endpoint_is_actionable():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(500, request=request))) as client:
            with pytest.raises(McpOAuthError, match="does not support dynamic client registration"):
                await register_client(client, "https://auth.test", client_metadata({}, "http://127.0.0.1/cb"),
                                      {"issuer": "https://auth.test"})

    run(scenario())


def test_registration_failure_reports_status_without_retry():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, content=b"invalid_redirect_uri", request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthRegistrationError) as error:
                await register_client(client, "https://auth.test", client_metadata({}, "http://127.0.0.1/cb"),
                                      {"registration_endpoint": "https://auth.test/register"})
        assert error.value.status == 400
        assert "invalid_redirect_uri" in str(error.value)
        assert len(requests) == 1

    run(scenario())


def test_callback_settings_keep_fixed_port_and_uri():
    fixed = callback_settings({"callback_port": 1456})
    assert fixed["fixed_redirect_url"] == "http://127.0.0.1:1456/callback"

    exact = callback_settings({"callback_url": "http://localhost:8080/oauth/callback"})
    assert exact["listen_host"] == "127.0.0.1"
    assert exact["fixed_redirect_url"] == "http://localhost:8080/oauth/callback"

    port_only = callback_settings({"callback_url": "http://127.0.0.1:8080"})
    assert port_only["path"] == "/callback"
    assert port_only["fixed_redirect_url"] == "http://127.0.0.1:8080/callback"

    dynamic = callback_settings({})
    assert dynamic["fixed_redirect_url"] is None
    assert dynamic["port"] is None
