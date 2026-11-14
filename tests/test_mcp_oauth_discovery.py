"""OAuth discovery parity: URL order, fallbacks, metadata validation (O02, O03)."""

import asyncio

import httpx
import pytest

from pi.coding_agent.mcp_oauth import (
    McpOAuthError,
    OAuthIssuerMismatchError,
    discover_authorization_server_metadata,
    discover_oauth_server_info,
    discover_protected_resource_metadata,
    select_resource,
)

AUTH_METADATA = {
    "issuer": "https://auth.test/tenant",
    "authorization_endpoint": "https://auth.test/tenant/authorize",
    "token_endpoint": "https://auth.test/tenant/token",
    "response_types_supported": ["code"],
    "code_challenge_methods_supported": ["S256"],
}


def run(coro):
    return asyncio.run(coro)


def test_protected_resource_falls_back_to_root_for_path_servers():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if str(request.url).endswith("/api/mcp"):
            return httpx.Response(404, request=request)
        return httpx.Response(200, json={
            "resource": "https://example.test/api/mcp",
            "authorization_servers": ["https://auth.test"],
        }, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            metadata = await discover_protected_resource_metadata("https://example.test/api/mcp", client)
        assert metadata["resource"] == "https://example.test/api/mcp"
        assert seen == [
            "https://example.test/.well-known/oauth-protected-resource/api/mcp",
            "https://example.test/.well-known/oauth-protected-resource",
        ]

    run(scenario())


def test_explicit_resource_metadata_url_is_not_replaced():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"resource": "https://example.test/mcp"}, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await discover_protected_resource_metadata(
                "https://example.test/mcp", client, "https://metadata.test/resource")
        assert seen == ["https://metadata.test/resource"]

    run(scenario())


def test_authorization_server_candidate_order_with_path_issuer():
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append(url)
        if url == "https://auth.test/.well-known/oauth-authorization-server/tenant":
            return httpx.Response(404, request=request)
        if url == "https://auth.test/.well-known/openid-configuration/tenant":
            return httpx.Response(502, request=request)
        return httpx.Response(200, json=AUTH_METADATA, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            metadata = await discover_authorization_server_metadata("https://auth.test/tenant", client)
        assert metadata["issuer"] == "https://auth.test/tenant"
        assert seen == [
            "https://auth.test/.well-known/oauth-authorization-server/tenant",
            "https://auth.test/.well-known/openid-configuration/tenant",
            "https://auth.test/tenant/.well-known/openid-configuration",
        ]

    run(scenario())


def test_500_is_not_a_discovery_miss():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(500, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(McpOAuthError, match="HTTP 500"):
                await discover_authorization_server_metadata("https://auth.test/tenant", client)
        assert len(seen) == 1

    run(scenario())


def test_issuer_mismatch_is_rejected():
    metadata = {**AUTH_METADATA, "issuer": "https://other.test"}

    def handler(request):
        return httpx.Response(200, json=metadata, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthIssuerMismatchError):
                await discover_authorization_server_metadata("https://auth.test/tenant", client)

    run(scenario())


def test_explicit_metadata_url_is_trusted_without_issuer_check():
    def handler(request):
        return httpx.Response(200, json={**AUTH_METADATA, "issuer": "https://other.test"}, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            info = await discover_oauth_server_info(
                "https://example.test/mcp", client,
                authorization_server_metadata_url="https://metadata.test/auth.json")
        assert info["authorization_server_url"] == "https://other.test"

    run(scenario())


@pytest.mark.parametrize("server,resource", [
    ("https://example.test/mcp", "https://other.test/mcp"),
    ("https://example.test/mcp", "https://example.test/api"),
])
def test_resource_must_cover_the_mcp_server(server, resource):
    with pytest.raises(McpOAuthError, match="does not match"):
        select_resource(server, {"resource": resource})


def test_resource_path_prefix_accepts_nested_server():
    assert select_resource("https://example.test/api/mcp", {"resource": "https://example.test/api"}) \
        == "https://example.test/api"


def test_broken_json_metadata_is_rejected():
    def handler(request):
        return httpx.Response(200, content=b"{not json", request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await discover_authorization_server_metadata("https://auth.test", client)

    run(scenario())


def test_protected_resource_metadata_rejects_unsafe_urls():
    def handler(request):
        return httpx.Response(200, json={"resource": "javascript:alert(1)"}, request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError, match="Invalid"):
                await discover_protected_resource_metadata("https://example.test/mcp", client)

    run(scenario())
