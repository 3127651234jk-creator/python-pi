"""MCP JSON-RPC clients for stdio and Streamable HTTP.

Adds on top of direct tool exposure: OAuth for HTTP servers (PKCE code flow
with a loopback callback, tokens persisted in ``mcp-auth.json``), resource
listing/reading/template tools with pagination, an independent background
notification pump per client (``notifications/tools/list_changed`` refresh,
``notifications/message`` logged to ``mcp.log``), one-shot automatic reconnect
on dropped connections, and lazy tool discovery.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import signal as process_signal
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from ..agent.types import AgentTool, AgentToolResult
from ..ai.models import ModelError
from ..ai.types import TextContent, ImageContent
from .config import Configuration, read_json, write_json
from .model_registry import expand, validate_url

RESOURCE_TOOLS = ("list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource")

JSON_RPC_METHOD_NOT_FOUND = -32601


class McpMethodNotFound(RuntimeError):
    """The server answered a request with JSON-RPC error -32601."""


def is_mcp_app_resource(item: dict) -> bool:
    """MCP App user interfaces, which only hosts that render them can use."""

    uri = item.get("uri") or item.get("uriTemplate") or ""
    mime = item.get("mimeType") or ""
    return uri.startswith("ui://") or re.search(r";\s*profile\s*=\s*\"?mcp-app\"?", mime, re.I) is not None


def listed_resource(server: str, item: dict) -> dict:
    """A listed resource or template without ``_meta`` and icons, tagged with its server."""

    return {"server": server, **{key: value for key, value in item.items() if key not in ("_meta", "icons")}}


class McpOAuthStore:
    """Persisted OAuth tokens for HTTP MCP servers, keyed by "<namespace>|<url>"."""

    def __init__(self, agent_dir: Path) -> None:
        self.path = agent_dir / "mcp-auth.json"

    def get(self, key: str) -> dict:
        return read_json(self.path).get(key, {})

    def set(self, key: str, record: dict) -> None:
        data = read_json(self.path)
        data[key] = record
        write_json(self.path, data)

    def remove(self, key: str) -> None:
        data = read_json(self.path)
        data.pop(key, None)
        write_json(self.path, data)


def oauth_key(namespace: str, url: str) -> str:
    return namespace + "|" + url


async def oauth_refresh(oauth: dict, credential: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(oauth["token_url"], data={
            "grant_type": "refresh_token", "client_id": oauth["client_id"],
            "refresh_token": credential["refresh"],
        })
    if response.status_code >= 400:
        raise RuntimeError(f"MCP OAuth refresh failed (HTTP {response.status_code})")
    data = response.json()
    access, expires = data.get("access_token"), data.get("expires_in", 3600)
    if not isinstance(access, str) or not access:
        raise RuntimeError("MCP OAuth refresh returned no access token")
    return {"access": access, "refresh": data.get("refresh_token", credential.get("refresh", "")),
            "expires": time.time() + float(expires)}


async def oauth_login(oauth: dict, notify, url: str) -> dict:
    """Interactive authorization-code + PKCE flow with a loopback callback."""
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    port = int(oauth.get("callback_port", 0))
    path = "/callback"
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    async def callback(reader, writer):
        try:
            request = await asyncio.wait_for(reader.readline(), 10)
            parts = request.decode("ascii", "replace").split(" ")
            target = urlsplit(parts[1]) if len(parts) == 3 else None
            params = parse_qs(target.query) if target else {}
            code = params.get("code", [""])[0]
            returned_state = params.get("state", [""])[0]
            if target and target.path == path and code and returned_state == state and not future.done():
                future.set_result(code)
                writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 18\r\n\r\nReturn to terminal")
            else:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        except (ValueError, asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(callback, "127.0.0.1", port)
    try:
        actual_port = server.sockets[0].getsockname()[1]
        redirect = f"http://127.0.0.1:{actual_port}{path}"
        state = secrets.token_urlsafe(16)
        authorize = oauth["authorize_url"] + ("&" if "?" in oauth["authorize_url"] else "?") + urlencode({
            "response_type": "code", "client_id": oauth["client_id"],
            "redirect_uri": redirect, "scope": oauth.get("scope", ""),
            "state": state, "code_challenge": challenge,
            "code_challenge_method": "S256", "resource": url,
        })
        notify(f"MCP authorization required for {url}\nOpen: {authorize}")
        code = await asyncio.wait_for(future, float(oauth.get("timeout", 600)))
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(oauth["token_url"], data={
                "grant_type": "authorization_code", "client_id": oauth["client_id"], "code": code,
                "redirect_uri": redirect, "code_verifier": verifier})
        if response.status_code >= 400:
            raise RuntimeError(f"MCP OAuth exchange failed (HTTP {response.status_code})")
        data = response.json()
        access = data.get("access_token")
        if not isinstance(access, str) or not access:
            raise RuntimeError("MCP OAuth exchange returned no access token")
        return {"access": access, "refresh": data.get("refresh_token", ""),
                "expires": time.time() + float(data.get("expires_in", 3600))}
    finally:
        server.close()
        await server.wait_closed()


class McpClient:
    def __init__(self, name: str, config: dict, cwd: Path) -> None:
        self.name, self.config, self.cwd = name, config, cwd
        self.process = None
        self.http = None
        self.session_id = None
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.pending: Dict[int, asyncio.Future] = {}
        self.pump_task: Optional[asyncio.Task] = None
        self.stderr_task = None
        self.tools = []
        self.tools_changed = False
        self.resources_changed = False
        self.resources_supported = False
        self.on_tools_changed = None  # sync/async callback, set by the manager
        self.on_log_message = None  # callback(server, params) for notifications/message
        self.lazy = bool(config.get("lazy", False))

    def bearer(self) -> str:
        oauth = self.config.get("oauth")
        if not oauth:
            return ""
        return self.credential.get("access", "") if getattr(self, "credential", None) else ""

    async def connect(self, auth_header: str = ""):
        if "command" in self.config:
            env = {**os.environ, **{k: expand(v) for k, v in self.config.get("env", {}).items()}}
            self.process = await asyncio.create_subprocess_exec(
                os.path.expanduser(self.config["command"]), *self.config.get("args", []),
                cwd=str(self.cwd / self.config.get("cwd", ".")), env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                start_new_session=sys.platform != "win32",
            )
            self.pump_task = asyncio.create_task(self._stdio_pump())
            async def drain():
                while await self.process.stderr.read(4096):
                    pass
            self.stderr_task = asyncio.create_task(drain())
        else:
            headers = {"Accept": "application/json, text/event-stream",
                **{k: expand(v) for k, v in self.config.get("headers", {}).items()}}
            if auth_header:
                headers["Authorization"] = "Bearer " + auth_header
            self.http = httpx.AsyncClient(timeout=self.config.get("timeout", 60), headers=headers)
        try:
            result = await self.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "pi-python", "version": "0.1.0"}})
        except BaseException:
            await self._stop_pump()
            raise
        self.version = result.get("protocolVersion", "2025-03-26")
        capabilities = result.get("capabilities", {}) or {}
        self.resources_supported = isinstance(capabilities, dict) and "resources" in capabilities
        await self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        if self.http is not None:
            self.pump_task = asyncio.create_task(self._http_listener())
        if not self.lazy:
            await self.list_tools()

    async def _stop_pump(self) -> None:
        if self.pump_task is not None:
            self.pump_task.cancel()
            try:
                await self.pump_task
            except (asyncio.CancelledError, Exception):
                pass
            self.pump_task = None
        self._fail_pending("MCP server disconnected")

    def _fail_pending(self, reason: str) -> None:
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

    async def _stdio_pump(self) -> None:
        """Independent notification loop: reads every server message as it arrives,
        so notifications are handled even when no request is in flight."""

        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    self._fail_pending("MCP server disconnected")
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError):
            self._fail_pending("MCP server disconnected")

    async def _http_listener(self) -> None:
        """Best-effort SSE listen stream for server push on Streamable HTTP."""

        try:
            headers = {"Accept": "text/event-stream", "MCP-Protocol-Version": getattr(self, "version", "2025-03-26")}
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            if self.bearer():
                headers["Authorization"] = "Bearer " + self.bearer()
            async with self.http.stream("GET", self.config["url"], headers=headers) as response:
                if response.status_code >= 400:
                    return
                data = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        payload = "\n".join(data)
                        data = []
                        try:
                            event = json.loads(payload)
                        except ValueError:
                            continue
                        if isinstance(event, dict):
                            await self._dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # the listen stream is optional per spec
            return

    async def _dispatch(self, message: dict) -> None:
        method = message.get("method")
        if method is not None:
            if "id" in message:
                await self._answer_server_request(message)
            else:
                self._handle_notification(message)
            return
        future = self.pending.pop(message.get("id"), None)
        if future is not None and not future.done():
            future.set_result(message)

    async def _answer_server_request(self, message: dict) -> None:
        if message.get("method") == "ping":
            await self.send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            await self.send({"jsonrpc": "2.0", "id": message["id"],
                             "error": {"code": JSON_RPC_METHOD_NOT_FOUND, "message": "Unsupported client request"}})

    def _handle_notification(self, message: dict) -> None:
        method = message.get("method")
        if method == "notifications/tools/list_changed":
            self.tools_changed = True
            if self.on_tools_changed is not None:
                asyncio.ensure_future(_maybe_call(self.on_tools_changed, self.name))
        elif method == "notifications/resources/list_changed":
            self.resources_changed = True
        elif method == "notifications/message" and self.on_log_message is not None:
            try:
                self.on_log_message(self.name, message.get("params") or {})
            except Exception:  # noqa: BLE001
                pass

    async def list_tools(self):
        cursor = None
        seen = set()
        tools = []
        while True:
            page = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(page.get("tools", []))
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if cursor in seen:
                raise ValueError("MCP repeated pagination cursor")
            seen.add(cursor)
        self.tools = tools

    async def _page(self, method: str, key: str, cursor):
        try:
            result = await self.request(method, {"cursor": cursor} if cursor else {})
        except McpMethodNotFound:
            return {key: []}
        if not isinstance(result, dict):
            return {key: []}
        return {key: result.get(key, []), "nextCursor": result.get("nextCursor")}

    async def _list_all(self, method: str, key: str):
        items, cursor, seen = [], None, set()
        while True:
            page = await self._page(method, key, cursor)
            items.extend(page[key])
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if cursor in seen:
                raise ValueError("MCP repeated pagination cursor")
            seen.add(cursor)
        return items

    async def resources_page(self, cursor=None):
        return await self._page("resources/list", "resources", cursor)

    async def list_resources(self):
        return await self._list_all("resources/list", "resources")

    async def resource_templates_page(self, cursor=None):
        return await self._page("resources/templates/list", "resourceTemplates", cursor)

    async def list_resource_templates(self):
        return await self._list_all("resources/templates/list", "resourceTemplates")

    async def read_resource(self, uri: str):
        return await self.request("resources/read", {"uri": uri})

    async def send(self, message):
        if self.process:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            await self.process.stdin.drain()
            return None
        headers = {"MCP-Protocol-Version": getattr(self, "version", "2025-03-26")}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.bearer():
            headers["Authorization"] = "Bearer " + self.bearer()
        async with self.http.stream("POST", self.config["url"], json=message, headers=headers) as response:
            if response.status_code == 401 or response.status_code == 403:
                raise RuntimeError(f"MCP authorization required (HTTP {response.status_code})")
            if response.status_code >= 400:
                raise RuntimeError(f"MCP HTTP {response.status_code}")
            self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
            if "id" not in message:
                return None
            if "text/event-stream" not in response.headers.get("content-type", ""):
                return json.loads(await response.aread())
            data = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    event = json.loads("\n".join(data))
                    data = []
                    if event.get("id") == message["id"]:
                        return event
                    if "method" in event and "id" not in event:
                        self._handle_notification(event)
            raise RuntimeError("MCP stream ended without a response")

    def _reply_result(self, method: str, reply):
        if not isinstance(reply, dict):
            raise RuntimeError("Invalid MCP response")
        if "error" in reply:
            error = reply.get("error") or {}
            if error.get("code") == JSON_RPC_METHOD_NOT_FOUND:
                raise McpMethodNotFound(f"MCP server does not implement {method}")
            raise RuntimeError(f"MCP request failed: {method}: {error.get('message') or method}")
        return reply.get("result", {})

    async def request(self, method, params):
        timeout = self.config.get("timeout", 60)
        if self.process is None:
            async with self.lock:
                self.sequence += 1
                message = {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}
                reply = await asyncio.wait_for(self.send(message), timeout)
                return self._reply_result(method, reply)
        async with self.lock:
            self.sequence += 1
            message = {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}
            future = asyncio.get_running_loop().create_future()
            self.pending[self.sequence] = future
            try:
                await self.send(message)
                reply = await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(f"MCP request timed out: {method}") from None
            finally:
                self.pending.pop(self.sequence, None)
        return self._reply_result(method, reply)

    async def close(self):
        await self._stop_pump()
        if self.http:
            await self.http.aclose()
            self.http = None
        if self.process:
            if self.process.returncode is None:
                if sys.platform == "win32":
                    killer = await asyncio.create_subprocess_exec("taskkill", "/F", "/T", "/PID", str(self.process.pid),
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    await killer.wait()
                else:
                    try:
                        os.killpg(self.process.pid, process_signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(self.process.wait(), 3)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None
            if self.stderr_task:
                self.stderr_task.cancel()
                self.stderr_task = None


class McpManager:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.clients = {}
        self.errors = {}
        self.tools = []
        self.oauth_store = McpOAuthStore(config.agent_dir)
        self.notify = print
        self.on_tools_changed = None  # set by the CLI to refresh agent.tools
        self.lazy_servers = set()
        self.log_path = config.agent_dir / "mcp.log"
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self.extension_servers: Dict[str, dict] = {}
        self.extension_server_owners: Dict[str, str] = {}

    def _log_message(self, server: str, params: dict) -> None:
        """Append one ``notifications/message`` to mcp.log; continuation lines are indented."""

        level = params.get("level", "info") if isinstance(params, dict) else "info"
        data = params.get("data") if isinstance(params, dict) else params
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = "\n".join(("    " + line) for line in text.splitlines()[1:]) if isinstance(text, str) else ""
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{timestamp} [{level}] {server}: {text.splitlines()[0] if isinstance(text, str) else text}\n")
                if lines:
                    handle.write(lines + "\n")
        except OSError:
            pass

    async def reload(self):
        await self.close()
        self.errors, self.tools = {}, []
        self._refresh_locks = {}
        servers = {}
        paths = [self.config.agent_dir / "mcp.json"]
        if self.config.project_trusted:
            paths.append(self.config.cwd / ".pi/mcp.json")
        for path in paths:
            entries = read_json(path).get("mcpServers", {})
            if not isinstance(entries, dict):
                self.errors[str(path)] = "mcpServers must be an object"
                continue
            for name, value in entries.items():
                if not isinstance(value, dict):
                    self.errors[name] = "Server configuration must be an object"
                    continue
                servers[name] = {**servers.get(name, {}), **value}
        # File configuration has precedence over ephemeral extension registrations.
        servers = {**self.extension_servers, **servers}
        normalized = set()
        for name, settings in servers.items():
            client = None
            try:
                namespace = re.sub(r"[^A-Za-z0-9_]", "_", name)
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or namespace in normalized:
                    raise ValueError("Invalid or colliding server name")
                normalized.add(namespace)
                if not settings.get("enabled", True):
                    self.errors[name] = "disabled"
                    continue
                if ("command" in settings) == ("url" in settings):
                    raise ValueError("Specify either command or url")
                if settings.get("type") not in (None, "stdio", "http", "streamable-http"):
                    raise ValueError("Unsupported MCP transport")
                timeout = settings.get("timeout", 60)
                if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
                    raise ValueError("timeout must be between 0 and 86400 seconds")
                if "command" in settings and (not isinstance(settings["command"], str) or not isinstance(settings.get("args", []), list) or not all(isinstance(arg, str) for arg in settings.get("args", []))):
                    raise ValueError("command must be an executable and args a string list")
                if "url" in settings:
                    validate_url(settings["url"])
                oauth = settings.get("oauth")
                if oauth is not None:
                    if not isinstance(oauth, dict) or not all(isinstance(oauth.get(k), str) and oauth[k]
                                                              for k in ("authorize_url", "token_url", "client_id")):
                        raise ValueError("MCP oauth requires authorize_url, token_url and client_id")
                    for key in ("authorize_url", "token_url"):
                        validate_url(oauth[key])
                client = McpClient(name, settings, self.config.cwd)
                if client.lazy and "url" not in settings:
                    raise ValueError("lazy tool discovery requires an HTTP server")
                await self._connect_client(client, namespace)
                self.clients[name] = client
                if client.lazy:
                    self.lazy_servers.add(name)
                else:
                    self._register_server_tools(namespace, client)
                if client.tools_changed:
                    # A list_changed notification raced the connect; list again now that
                    # the client is registered.
                    await self.ensure_tools(name)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, ModelError,
                    asyncio.TimeoutError, httpx.HTTPError) as exc:
                self.errors[name] = str(exc) or type(exc).__name__
                if client:
                    await client.close()
                self.clients.pop(name, None)
                self.tools = [tool for tool in self.tools if not getattr(tool, "_mcp_server", None) == name]
        self._register_resource_tools()

    async def register_extension_server(self, name: str, settings: dict, owner: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid MCP server name")
        if not isinstance(settings, dict) or ("command" in settings) == ("url" in settings):
            raise ValueError("MCP server must specify either command or url")
        existing = self.extension_server_owners.get(name)
        if existing is not None and existing != owner:
            raise ValueError(f'MCP server "{name}" is already registered by extension "{existing}"')
        namespace = re.sub(r"[^A-Za-z0-9_]", "_", name)
        for other in self.extension_servers:
            if other != name and re.sub(r"[^A-Za-z0-9_]", "_", other) == namespace:
                raise ValueError(f'MCP server "{name}" conflicts with registered server "{other}"')
        self.extension_servers[name] = dict(settings)
        self.extension_server_owners[name] = owner
        await self.reload()
        if self.on_tools_changed is not None:
            await _maybe_call(self.on_tools_changed, name)

    async def unregister_extension_server(self, name: str, owner: str) -> None:
        existing = self.extension_server_owners.get(name)
        if existing is None:
            return
        if existing != owner:
            raise ValueError(f'MCP server "{name}" belongs to extension "{existing}"')
        self.extension_server_owners.pop(name, None)
        self.extension_servers.pop(name, None)
        await self.reload()
        if self.on_tools_changed is not None:
            await _maybe_call(self.on_tools_changed, name)

    def registered_extension_servers(self) -> List[dict]:
        return [{"name": name, "config": dict(config),
                 "extensionPath": self.extension_server_owners[name]}
                for name, config in self.extension_servers.items()]

    async def _connect_client(self, client: McpClient, namespace: str) -> None:
        """Connect with OAuth credentials when configured; never auto-launches a browser."""
        client.credential = None
        client.on_log_message = self._log_message
        client.on_tools_changed = self._handle_tools_changed
        auth_header = ""
        oauth = client.config.get("oauth")
        if oauth and "url" in client.config:
            key = oauth_key(namespace, client.config["url"])
            record = self.oauth_store.get(key)
            if record.get("access"):
                client.credential = record
                if record.get("expires", 0) <= time.time() + 60:
                    if record.get("refresh"):
                        record = await oauth_refresh(oauth, record)
                        self.oauth_store.set(key, record)
                        client.credential = record
                    else:
                        raise RuntimeError("MCP OAuth token expired; run /mcp login")
                auth_header = record["access"]
            else:
                client.needs_login = True
        await client.connect(auth_header)

    async def _handle_tools_changed(self, server: str) -> None:
        """Refresh one server's tools after notifications/tools/list_changed. A server that
        is not registered yet is still connecting; reload lists its tools anyway."""

        if server not in self.clients:
            return
        try:
            await self.ensure_tools(server)
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
            self.errors[server] = f"Failed to refresh tools: {exc}"

    def _register_server_tools(self, namespace: str, client: McpClient) -> None:
        for tool in client.tools:
            raw = tool["name"]
            safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
            suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
            tool_name = f"mcp__{namespace}__{safe}_{suffix}"
            self.tools.append(self._make_tool(tool_name, tool.get("description", raw),
                tool.get("inputSchema", {"type": "object"}), client.name, raw))

    def _make_tool(self, tool_name: str, description: str, parameters: dict, server: str, raw: str) -> AgentTool:
        async def execute(call_id, params, signal=None, on_update=None):
            if signal:
                signal.throw_if_aborted()
            result = await self.call(server, "tools/call", {"name": raw, "arguments": params}, signal)
            content = []
            for block in result.get("content", []):
                if block.get("type") == "text":
                    content.append(TextContent(text=block.get("text", "")))
                elif block.get("type") == "image":
                    content.append(ImageContent(data=block["data"], mime_type=block["mimeType"]))
                else:
                    content.append(TextContent(text=json.dumps(block, ensure_ascii=False)))
            return AgentToolResult(content=content or [TextContent(text=json.dumps(result))],
                structured_content=result.get("structuredContent"), is_error=bool(result.get("isError")))
        tool = AgentTool(name=tool_name, description=description, parameters=parameters, execute=execute)
        tool._mcp_server = server
        return tool

    def _register_resource_tools(self) -> None:
        """The resource tools Codex and opencode use: list, list templates, read. With a
        ``server`` argument one page is listed and ``cursor`` continues it; without, every
        page of every server, with per-server errors collected."""

        if not any(client.resources_supported for client in self.clients.values()):
            return

        def find_server(name):
            client = self.clients.get(name)
            if client is None or not client.resources_supported:
                available = ", ".join(sorted(name for name, other in self.clients.items()
                                             if other.resources_supported))
                raise ValueError(f'MCP server "{name}" has no resources'
                                 + (f". Servers with resources: {available}" if available else ""))
            return client

        async def list_call(call_id, params, signal=None, on_update=None, key="resources", method="resources/list"):
            if signal:
                signal.throw_if_aborted()
            data = params or {}
            server_name, cursor = data.get("server"), data.get("cursor")
            if server_name:
                client = find_server(server_name)
                page = await (client.resources_page(cursor) if method == "resources/list"
                              else client.resource_templates_page(cursor))
                payload = {"server": client.name,
                           key: [listed_resource(client.name, item) for item in page[key]
                                 if not is_mcp_app_resource(item)]}
                if page.get("nextCursor"):
                    payload["nextCursor"] = page["nextCursor"]
            else:
                if cursor:
                    raise ValueError("cursor can only be used when a server is specified")
                payload = {key: [], "errors": []}
                for name in sorted(self.clients):
                    client = self.clients[name]
                    if not client.resources_supported:
                        continue
                    try:
                        if method == "resources/list":
                            items = await client.list_resources()
                        else:
                            items = await client.list_resource_templates()
                        payload[key].extend(listed_resource(name, item) for item in items
                                            if not is_mcp_app_resource(item))
                    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                        payload["errors"].append({"server": name, "error": str(exc)})
                if not payload["errors"]:
                    del payload["errors"]
            return AgentToolResult(
                content=[TextContent(text=json.dumps(payload, ensure_ascii=False))], details={},
                structured_content=payload)

        async def read_resource(call_id, params, signal=None, on_update=None):
            if signal:
                signal.throw_if_aborted()
            data = params or {}
            name, uri = data.get("server"), data.get("uri")
            if not name or not uri:
                raise ValueError("read_mcp_resource requires server and uri")
            client = find_server(name)
            result = await self.call(name, "resources/read", {"uri": uri}, signal)
            content = []
            blocks = result.get("contents", []) if isinstance(result, dict) else []
            payload_contents = []
            for block in blocks:
                payload_contents.append({k: v for k, v in block.items() if k != "_meta"})
                if len(blocks) > 1:
                    content.append(TextContent(text=f"{block.get('uri')}:"))
                if block.get("text") is not None:
                    content.append(TextContent(text=block["text"]))
                elif block.get("blob") is not None:
                    saved = self._save_blob(block.get("blob"), block.get("mimeType") or "")
                    content.append(TextContent(text=saved))
                else:
                    content.append(TextContent(text=json.dumps(block, ensure_ascii=False)))
            payload = {"server": name, "uri": uri, "contents": payload_contents}
            if not content:
                content.append(TextContent(text=f"Resource {uri} is empty."))
            return AgentToolResult(content=content, details={}, structured_content=payload)

        list_params = {"type": "object",
                       "properties": {"server": {"type": "string", "description": "MCP server name. Omit to list every server with resources."},
                                      "cursor": {"type": "string", "description": "Opaque cursor from a previous call with the same server; omit for the first page."}},
                       "required": []}
        self.tools.append(AgentTool(name="list_mcp_resources",
            description="Lists resources provided by MCP servers. Resources allow servers to share data that provides "
                        "context to language models, such as files, database schemas, or application-specific information.",
            parameters=list_params, execute=lambda *args, **kwargs: list_call(*args, key="resources", method="resources/list", **kwargs)))
        self.tools.append(AgentTool(name="list_mcp_resource_templates",
            description="Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to "
                        "share data that takes parameters and provides context to language models.",
            parameters=list_params, execute=lambda *args, **kwargs: list_call(*args, key="resourceTemplates", method="resources/templates/list", **kwargs)))
        self.tools.append(AgentTool(name="read_mcp_resource",
            description="Read a specific resource from an MCP server given the server name and resource URI.",
            parameters={"type": "object",
                        "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
                        "required": ["server", "uri"]},
            execute=read_resource))

    def _save_blob(self, blob: str, mime_type: str) -> str:
        """Save a base64 resource blob to a temp file; returns the model-facing text."""

        try:
            data = base64.b64decode(blob)
        except (ValueError, TypeError):
            return "[Resource blob could not be decoded]"
        suffix = ".bin"
        if mime_type == "image/png":
            suffix = ".png"
        elif mime_type == "image/jpeg":
            suffix = ".jpg"
        elif mime_type == "image/gif":
            suffix = ".gif"
        elif mime_type == "image/webp":
            suffix = ".webp"
        elif mime_type in ("text/plain", "text/markdown"):
            suffix = ".txt" if mime_type == "text/plain" else ".md"
        descriptor, path = tempfile.mkstemp(prefix="pi-mcp-resource-", suffix=suffix)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        return f"[Resource saved to {path} ({mime_type or 'unknown type'}, {len(data)} bytes)]"

    async def call(self, server: str, method: str, params: dict, signal=None):
        """Route a request through one client, reconnecting once on a dropped connection."""
        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        try:
            return await client.request(method, params)
        except (RuntimeError, httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            message = str(exc)
            transient = "disconnected" in message or "MCP HTTP" in message or isinstance(exc, httpx.HTTPError) \
                or isinstance(exc, (asyncio.TimeoutError, OSError))
            if not transient:
                raise
            await self.reconnect(server)
            if signal:
                signal.throw_if_aborted()
            return await self.clients[server].request(method, params)

    async def reconnect(self, server: str) -> None:
        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        namespace = re.sub(r"[^A-Za-z0-9_]", "_", server)
        replacement = McpClient(server, client.config, client.cwd)
        try:
            await self._connect_client(replacement, namespace)
        except BaseException:
            try:
                await replacement.close()
            except BaseException:
                pass
            raise
        # Retire the old client only after the replacement connected: a failed
        # reconnect must not leave a closed client registered, where every
        # later call would raise AttributeError and never retry.
        await client.close()
        self.clients[server] = replacement
        await self.ensure_tools(server)

    async def ensure_tools(self, server: str) -> None:
        """Discover tools for a lazily-connected server and refresh changed tool sets."""

        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        lock = self._refresh_locks.setdefault(server, asyncio.Lock())
        async with lock:
            namespace = re.sub(r"[^A-Za-z0-9_]", "_", server)
            before = len(self.tools)
            self.tools = [tool for tool in self.tools if getattr(tool, "_mcp_server", None) != server]
            if client.tools_changed or not client.tools:
                await client.list_tools()
                client.tools_changed = False
            self._register_server_tools(namespace, client)
            self.lazy_servers.discard(server)
            if self.on_tools_changed is not None and len(self.tools) != before:
                await _maybe_call(self.on_tools_changed)

    async def refresh_changed(self) -> bool:
        """Re-list tools for servers that reported notifications/tools/list_changed."""
        changed = [name for name, client in self.clients.items() if client.tools_changed]
        for name in changed:
            await self.ensure_tools(name)
        return bool(changed)

    async def oauth_login(self, server: str) -> None:
        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        oauth = client.config.get("oauth")
        if not oauth or "url" not in client.config:
            raise ValueError(f"MCP server has no OAuth configuration: {server}")
        record = await oauth_login(oauth, self.notify, client.config["url"])
        self.oauth_store.set(oauth_key(re.sub(r"[^A-Za-z0-9_]", "_", server), client.config["url"]), record)
        client.credential = record

    def oauth_logout(self, server: str) -> None:
        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        if "url" in client.config:
            self.oauth_store.remove(oauth_key(re.sub(r"[^A-Za-z0-9_]", "_", server), client.config["url"]))
            client.credential = None

    async def close(self):
        for client in self.clients.values():
            await client.close()
        self.clients = {}
        self.lazy_servers = set()


async def _maybe_call(fn, *args):
    result = fn(*args)
    if asyncio.iscoroutine(result):
        await result
