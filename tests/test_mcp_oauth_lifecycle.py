"""MCP OAuth lifecycle: discovery on 401, sign-in, refresh, logout (O01, O06-O11)."""

import asyncio
import json
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.mcp import McpManager, oauth_key
from pi.coding_agent.mcp_oauth import McpLoginRequired


RESOURCE_METADATA_URL = "https://example.test/.well-known/oauth-protected-resource/mcp"
AUTH_METADATA = {
    "issuer": "https://auth.test",
    "authorization_endpoint": "https://auth.test/authorize",
    "token_endpoint": "https://auth.test/token",
    "registration_endpoint": "https://auth.test/register",
    "response_types_supported": ["code"],
    "code_challenge_methods_supported": ["S256"],
    "token_endpoint_auth_methods_supported": ["none"],
}


class FakeOAuthMcp:
    """Routes the MCP endpoint, discovery documents, registration and token requests."""

    def __init__(self):
        self.requests = []
        self.token_requests = []
        self.access = "token-1"
        self.refresh = "refresh-1"
        self.refresh_count = 0
        self.expected = None  # path -> bearer, for multi-server tests
        # Created inside the event loop: asyncio.Event binds to the running loop on 3.9.
        self.token_started = None
        self.token_release = None  # asyncio.Event to delay the token response
        self.fail_refresh = False

    async def send(self, client, request, **kwargs):
        url = str(request.url)
        method = request.method
        body = request.content
        headers = dict(request.headers)
        self.requests.append((method, url, headers, body))
        if url.startswith("https://example.test/") and not url.split("example.test", 1)[1].startswith("/."):
            return self.mcp_response(request, headers, body)
        if url == RESOURCE_METADATA_URL:
            return httpx.Response(200, json={
                "resource": "https://example.test/mcp",
                "authorization_servers": ["https://auth.test"],
                "scopes_supported": ["tools.read"],
            }, request=request)
        if url == "https://auth.test/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=AUTH_METADATA, request=request)
        if url == "https://auth.test/register":
            payload = json.loads(body)
            return httpx.Response(200, json={
                "client_id": "issued-client",
                "redirect_uris": payload.get("redirect_uris", []),
                "token_endpoint_auth_method": payload.get("token_endpoint_auth_method", "none"),
            }, request=request)
        if url == "https://auth.test/token":
            return await self.token_response(request, body)
        if url.startswith("https://auth.test/.well-known/"):
            return httpx.Response(404, request=request)
        return httpx.Response(404, request=request)

    def mcp_response(self, request, headers, body):
        path = urlsplit(str(request.url)).path
        if self.expected is not None:
            required = self.expected.get(path)
        else:
            required = self.access
        if headers.get("authorization") != f"Bearer {required}":
            return httpx.Response(401, headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{RESOURCE_METADATA_URL}", scope="tools.read"',
            }, request=request)
        try:
            payload = json.loads(body)
        except ValueError:
            return httpx.Response(202, request=request)
        if "id" not in payload:
            return httpx.Response(202, request=request)
        method = payload.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {}}
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo",
                                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text",
                                   "text": payload["params"]["arguments"]["text"]}]}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
                              request=request)

    async def token_response(self, request, body):
        data = {key: values[0] for key, values in parse_qs(body.decode()).items()}
        self.token_requests.append(data)
        if self.token_started is None:
            self.token_started = asyncio.Event()
        self.token_started.set()
        if self.token_release is not None:
            await self.token_release.wait()
        if data.get("grant_type") == "refresh_token":
            if self.fail_refresh or data.get("refresh_token") != self.refresh:
                return httpx.Response(400, json={"error": "invalid_grant",
                                                 "error_description": "refresh token rejected"}, request=request)
            self.refresh_count += 1
            self.access = f"refreshed-{self.refresh_count}"
            self.refresh = f"rotated-{self.refresh_count}"
            return httpx.Response(200, json={
                "access_token": self.access, "token_type": "Bearer",
                "expires_in": 3600, "refresh_token": self.refresh,
            }, request=request)
        self.access = "access-code"
        self.refresh = "refresh-code"
        return httpx.Response(200, json={
            "access_token": self.access, "token_type": "Bearer",
            "expires_in": 3600, "refresh_token": self.refresh,
        }, request=request)


def install(monkeypatch, fake):
    async def send(self, request, **kwargs):
        return await fake.send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)


def oauth_config(tmp_path, oauth=None, name="remote", path="/mcp"):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    server = {"url": f"https://example.test{path}"}
    if oauth is not None:
        server["oauth"] = oauth
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {name: server}})
    return config


def record_for(tmp_path, access, refresh, expires=None, client=None):
    return {
        "access": access, "refresh": refresh, "token_type": "Bearer",
        "expires": time.time() + 3600 if expires is None else expires,
        "client": client or {"client_id": "issued-client", "redirect_uris": ["http://127.0.0.1/callback"]},
    }


async def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.01)


async def send_callback(redirect_url, query):
    parts = urlsplit(redirect_url)
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
    target = parts.path + "?" + urlencode(query)
    writer.write(f"GET {target} HTTP/1.1\r\nHost: {parts.netloc}\r\nConnection: close\r\n\r\n".encode())
    status = await reader.readline()
    await reader.read()
    writer.close()
    return status.decode()


def authorization_url(messages):
    for message in messages:
        if "Open: " in message:
            return message.split("Open: ", 1)[1].strip()
    raise AssertionError("no authorization URL was shown")


def test_o01_401_discovery_keeps_server_loginable(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)
    messages = []

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        manager.notify = messages.append
        await manager.reload()
        assert manager.clients == {}
        assert "remote" in manager.login_servers
        assert "requires sign-in" in manager.errors["remote"]
        assert not messages  # startup never starts or opens a browser
        urls = [url for _method, url, _headers, _body in fake.requests]
        assert RESOURCE_METADATA_URL in urls
        assert "https://auth.test/.well-known/oauth-authorization-server" in urls
        assert "https://auth.test/authorize" not in "".join(urls)
        await manager.close()

    asyncio.run(scenario())


def test_o06_full_login_then_tool_call(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)
    messages = []
    changed = []

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        manager.notify = messages.append
        manager.on_tools_changed = lambda server: changed.append(server)
        await manager.reload()
        assert manager.clients == {}

        login = asyncio.create_task(manager.oauth_login("remote"))
        await wait_for(lambda: any("Open: " in message for message in messages))
        url = authorization_url(messages)
        query = parse_qs(urlsplit(url).query)
        redirect = query["redirect_uri"][0]
        assert query["response_type"] == ["code"]
        assert query["client_id"] == ["issued-client"]
        assert query["code_challenge_method"] == ["S256"]
        assert query["scope"] == ["tools.read"]
        assert query["resource"] == ["https://example.test/mcp"]
        assert query["state"][0]

        registration = next(body for method, request_url, _headers, body in fake.requests
                            if request_url == "https://auth.test/register")
        registration = json.loads(registration)
        assert registration["application_type"] == "native"
        assert registration["redirect_uris"] == [redirect]
        assert registration["token_endpoint_auth_method"] == "none"

        status = await send_callback(redirect, {"code": "auth-code", "state": query["state"][0]})
        assert "200" in status
        await asyncio.wait_for(login, 5)

        assert "remote" in manager.clients
        assert manager.clients["remote"].bearer() == "access-code"
        assert manager.tools and manager.tools[0].name.startswith("mcp__remote__echo")
        assert "remote" in changed

        exchange = fake.token_requests[-1]
        assert exchange["grant_type"] == "authorization_code"
        assert exchange["code"] == "auth-code"
        assert exchange["redirect_uri"] == redirect
        assert exchange["client_id"] == "issued-client"
        assert exchange["resource"] == "https://example.test/mcp"
        assert exchange["code_verifier"]

        result = await manager.tools[0].execute("call", {"text": "hello"})
        assert result.content[0].text == "hello"
        assert any(url == "https://example.test/mcp" and headers.get("authorization") == "Bearer access-code"
                   for _method, url, headers, _body in fake.requests)
        await manager.close()

    asyncio.run(scenario())


def test_o07_rejected_authorization_saves_no_tokens(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)
    messages = []

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        manager.notify = messages.append
        await manager.reload()
        login = asyncio.create_task(manager.oauth_login("remote"))
        await wait_for(lambda: any("Open: " in message for message in messages))
        query = parse_qs(urlsplit(authorization_url(messages)).query)
        redirect = query["redirect_uri"][0]
        status = await send_callback(redirect, {"error": "access_denied",
                                                "error_description": "user denied",
                                                "state": query["state"][0]})
        assert "400" in status
        with pytest.raises(ValueError, match="denied"):
            await asyncio.wait_for(login, 5)
        key = oauth_key("remote", "https://example.test/mcp")
        record = manager.oauth_store.get(key)
        assert not record.get("access")
        assert not record.get("refresh")
        assert not fake.token_requests
        await manager.close()

    asyncio.run(scenario())


def test_o07_wrong_state_or_path_does_not_authorize(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)
    messages = []

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        manager.notify = messages.append
        await manager.reload()
        login = asyncio.create_task(manager.oauth_login("remote"))
        await wait_for(lambda: any("Open: " in message for message in messages))
        query = parse_qs(urlsplit(authorization_url(messages)).query)
        redirect = query["redirect_uri"][0]
        key = oauth_key("remote", "https://example.test/mcp")

        status = await send_callback(redirect.replace("/callback", "/wrong"), {
            "code": "evil", "state": query["state"][0]})
        assert "404" in status
        status = await send_callback(redirect, {"code": "evil", "state": "wrong-state"})
        assert "400" in status
        await asyncio.sleep(0)
        assert not login.done()
        assert not manager.oauth_store.get(key).get("access")
        assert not fake.token_requests

        status = await send_callback(redirect, {"code": "auth-code", "state": query["state"][0]})
        assert "200" in status
        await asyncio.wait_for(login, 5)
        assert fake.token_requests[-1]["code"] == "auth-code"
        await manager.close()

    asyncio.run(scenario())


def test_o07_login_timeout_closes_the_flow(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)
    messages = []

    async def scenario():
        manager = McpManager(oauth_config(tmp_path, {"timeout": 0.2}))
        manager.notify = messages.append
        await manager.reload()
        with pytest.raises(ValueError, match="timed out"):
            await asyncio.wait_for(manager.oauth_login("remote"), 5)
        assert not fake.token_requests
        await manager.close()

    asyncio.run(scenario())


def test_o08_expired_token_refresh_and_rotation(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    fake.refresh = "r1"
    install(monkeypatch, fake)

    async def scenario():
        config = oauth_config(tmp_path)
        manager = McpManager(config)
        key = oauth_key("remote", "https://example.test/mcp")
        manager.oauth_store.set(key, record_for(tmp_path, "old", "r1", expires=0))
        await manager.reload()
        assert manager.clients["remote"].bearer() == "refreshed-1"
        assert fake.token_requests[-1]["grant_type"] == "refresh_token"
        assert fake.token_requests[-1]["refresh_token"] == "r1"
        stored = manager.oauth_store.get(key)
        assert stored["access"] == "refreshed-1"
        assert stored["refresh"] == "rotated-1"

        # Two concurrent 401s share one refresh. Both the client and the store must
        # hold the rejected token; otherwise the store already has a newer grant.
        stale = {**manager.oauth_store.get(key), "access": "stale"}
        manager.oauth_store.set(key, stale)
        manager.clients["remote"].credential = stale
        before = len(fake.token_requests)
        results = await asyncio.gather(
            manager.call("remote", "tools/list", {}),
            manager.call("remote", "tools/list", {}))
        assert all(result["tools"][0]["name"] == "echo" for result in results)
        assert len(fake.token_requests) == before + 1
        assert fake.token_requests[-1]["refresh_token"] == "rotated-1"
        assert manager.oauth_store.get(key)["refresh"] == "rotated-2"
        await manager.close()

        # A restarted process continues from the rotated refresh token.
        fake.refresh = "rotated-2"
        fake.refresh_count = 2
        restarted = McpManager(config)
        restarted.oauth_store.set(key, record_for(tmp_path, "refreshed-2", "rotated-2", expires=0))
        await restarted.reload()
        assert fake.token_requests[-1]["refresh_token"] == "rotated-2"
        assert restarted.oauth_store.get(key)["refresh"] == "rotated-3"
        await restarted.close()

    asyncio.run(scenario())


def test_o09_close_waits_for_refresh_persistence(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    fake.refresh = "r1"
    install(monkeypatch, fake)

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        key = oauth_key("remote", "https://example.test/mcp")
        manager.oauth_store.set(key, record_for(tmp_path, "old", "r1", expires=0))
        persisted = asyncio.Event()
        release = asyncio.Event()
        original = manager._persist_oauth_record

        async def slow_persist(key_value, record, epoch, server):
            persisted.set()
            await release.wait()
            await original(key_value, record, epoch, server)

        manager._persist_oauth_record = slow_persist
        reload_task = asyncio.create_task(manager.reload())
        await asyncio.wait_for(persisted.wait(), 5)
        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0.05)
        assert not close_task.done()
        release.set()
        await asyncio.wait_for(close_task, 5)
        await asyncio.wait_for(reload_task, 5)
        stored = manager.oauth_store.get(key)
        assert stored["access"] == "refreshed-1"
        assert stored["refresh"] == "rotated-1"

    asyncio.run(scenario())


def test_o11_static_oauth_still_connects_without_discovery(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    fake.access = "static-token"
    install(monkeypatch, fake)
    oauth = {"authorize_url": "https://auth.test/authorize",
             "token_url": "https://auth.test/token",
             "client_id": "static-client"}

    async def scenario():
        manager = McpManager(oauth_config(tmp_path, oauth))
        key = oauth_key("remote", "https://example.test/mcp")
        manager.oauth_store.set(key, record_for(tmp_path, "static-token", "", expires=None))
        await manager.reload()
        assert manager.clients["remote"].bearer() == "static-token"
        assert not any("well-known" in url for _method, url, _headers, _body in fake.requests)
        await manager.close()

    asyncio.run(scenario())


def test_o11_credentials_are_isolated_per_server(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    fake.expected = {"/a": "token-a", "/b": "token-b"}
    install(monkeypatch, fake)
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "a": {"url": "https://example.test/a"},
        "b": {"url": "https://example.test/b"},
    }})

    async def scenario():
        manager = McpManager(config)
        manager.oauth_store.set(oauth_key("a", "https://example.test/a"),
                                record_for(tmp_path, "token-a", "refresh-a"))
        manager.oauth_store.set(oauth_key("b", "https://example.test/b"),
                                record_for(tmp_path, "token-b", "refresh-b"))
        await manager.reload()
        assert manager.clients["a"].bearer() == "token-a"
        assert manager.clients["b"].bearer() == "token-b"
        await manager.close()

    asyncio.run(scenario())


def test_o11_logout_is_not_undone_by_a_late_refresh(tmp_path, monkeypatch):
    fake = FakeOAuthMcp()
    install(monkeypatch, fake)

    async def scenario():
        manager = McpManager(oauth_config(tmp_path))
        key = oauth_key("remote", "https://example.test/mcp")
        manager.oauth_store.set(key, record_for(tmp_path, "expired", "r1", expires=0))
        fake.refresh = "r1"
        await manager.reload()
        client = manager.clients["remote"]
        stored = manager.oauth_store.get(key)
        stale = {**stored, "access": "stale"}
        manager.oauth_store.set(key, stale)
        client.credential = stale
        fake.refresh = stored["refresh"]
        fake.token_started = asyncio.Event()
        fake.token_release = asyncio.Event()

        call = asyncio.create_task(manager.call("remote", "tools/list", {}))
        await asyncio.wait_for(fake.token_started.wait(), 5)
        await manager.oauth_logout("remote")
        fake.token_release.set()
        with pytest.raises(McpLoginRequired):
            await asyncio.wait_for(call, 5)
        assert manager.oauth_store.get(key) == {}
        assert "remote" not in manager.clients
        await manager.close()

    asyncio.run(scenario())
