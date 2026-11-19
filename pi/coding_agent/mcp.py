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
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import sys
import signal as process_signal
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

import httpx

from ..agent.types import AgentTool, AgentToolResult
from ..ai.models import ModelError
from ..ai.types import TextContent, ImageContent
from .config import Configuration, read_json, write_json
from .mcp_oauth import (
    OAUTH_REQUEST_TIMEOUT_SECONDS,
    McpAuthorizationRequired,
    McpLoginRequired,
    McpOAuthError,
    OAuthIssuerMismatchError,
    callback_settings,
    client_metadata,
    discover_authorization_server_metadata,
    discover_oauth_server_info,
    exchange_authorization_code,
    merge_scopes,
    oauth_settings,
    parse_www_authenticate,
    record_from_tokens,
    refresh_authorization,
    register_client,
    select_resource,
    start_authorization,
    static_client,
    validate_oauth_settings,
)
from .model_registry import expand, validate_url
from .oauth_callback import start_oauth_callback_server
from ..util import AbortError

RESOURCE_TOOLS = ("list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource")

JSON_RPC_METHOD_NOT_FOUND = -32601


def tool_pattern_reg_exp(pattern: str) -> re.Pattern:
    """Regular expression for a tool name pattern where ``*`` matches any characters,
    as ``--tools`` and ``--exclude-tools`` accept them."""

    source = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.compile("^" + source + "$")


def create_tool_name_matcher(entries: Sequence[str]) -> Callable[[str], bool]:
    """Whether a tool name matches any of the entries, each an exact name or a pattern."""

    names = {entry for entry in entries if "*" not in entry}
    patterns = [tool_pattern_reg_exp(entry) for entry in entries if "*" in entry]
    return lambda name: name in names or any(pattern.match(name) for pattern in patterns)


def is_mcp_tool_name(name: str) -> bool:
    """Whether a tool comes from MCP: a server tool (``mcp__<server>__<tool>``) or a resource tool."""

    return name.startswith("mcp__") or name in RESOURCE_TOOLS


def _windows_batch_command(command: str, args: List[str]) -> str:
    """Escape both cmd.exe parses when a batch shim forwards arguments with %*."""
    if any("\r" in value or "\n" in value for value in [command, *args]):
        raise ValueError("stdio command and args must not contain CR or LF")
    meta = re.compile(r'([()\[\]%!^"`<>&|;, *?])')
    escaped = [meta.sub(r"^\1", command)]
    for arg in args:
        arg = re.sub(r'(\\*)"', lambda match: match[1] * 2 + '\\"', arg)
        trailing = len(arg) - len(arg.rstrip("\\"))
        arg += "\\" * trailing
        arg = meta.sub(r"^\1", '"' + arg + '"')
        arg = meta.sub(r"^\1", arg)
        escaped.append(arg)
    return " ".join(escaped)


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
        # Set by close(); a connect in progress checks it between stages and aborts
        # before doing any I/O on a transport the shutdown already closed.
        self.shutdown = asyncio.Event()
        # One teardown shared by every caller: a second close() must wait for the
        # first cleanup instead of returning early because the fields were cleared.
        self._close_task: Optional[asyncio.Task] = None

    @property
    def closed(self) -> bool:
        return self.shutdown.is_set()

    def bearer(self) -> str:
        # Automatic OAuth (a 401 without an ``oauth`` config block) still stores a
        # credential; servers configured with an Authorization header never get one.
        return self.credential.get("access", "") if getattr(self, "credential", None) else ""

    async def connect(self, auth_header: str = ""):
        if self.closed:
            raise RuntimeError(f"MCP server {self.name} is closed")
        if "command" in self.config:
            env = {**os.environ, **{k: expand(v) for k, v in self.config.get("env", {}).items()}}
            command = os.path.expanduser(self.config["command"])
            args = self.config.get("args", [])
            cwd = str(self.cwd / self.config.get("cwd", "."))
            if sys.platform == "win32":
                candidate = os.path.join(cwd, command) if os.path.dirname(command) else command
                command = shutil.which(candidate, path=env.get("PATH")) or command
            options = dict(
                cwd=cwd, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                start_new_session=sys.platform != "win32",
            )
            if sys.platform == "win32" and os.path.splitext(command)[1].lower() in (".cmd", ".bat"):
                self.process = await asyncio.create_subprocess_shell(
                    _windows_batch_command(command, args), executable=env.get("COMSPEC", "cmd.exe"), **options)
            else:
                self.process = await asyncio.create_subprocess_exec(command, *args, **options)
            if self.closed:
                # close() ran while the process was being spawned; tear it down
                # instead of leaving a live server nobody will talk to.
                await self.close()
                raise RuntimeError(f"MCP server {self.name} is closed")
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
            if self.closed:
                await self.close()
                raise RuntimeError(f"MCP server {self.name} is closed")
        try:
            result = await self.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "pi-python", "version": "0.1.0"}})
        except BaseException:
            # A failed or cancelled connect must not leave the transport open:
            # cancelling the pump alone keeps a spawned stdio server running.
            await self.close()
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
        """Listen without an idle read deadline; retry interrupted push streams."""
        delay = 1.0
        timeout = httpx.Timeout(self.config.get("timeout", 60), read=None)
        while True:
            try:
                headers = {"Accept": "text/event-stream", "MCP-Protocol-Version": getattr(self, "version", "2025-03-26")}
                if self.session_id:
                    headers["Mcp-Session-Id"] = self.session_id
                if self.bearer():
                    headers["Authorization"] = "Bearer " + self.bearer()
                async with self.http.stream("GET", self.config["url"], headers=headers, timeout=timeout) as response:
                    if response.status_code == 405:
                        return  # Servers may explicitly decline the optional GET stream.
                    response.raise_for_status()
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
                            await self._dispatch(event)
                            delay = 1.0
                logging.getLogger(__name__).warning("MCP %s push stream ended; reconnecting", self.name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep optional push failures isolated
                logging.getLogger(__name__).warning("MCP %s push stream failed: %s", self.name, exc)
                if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500:
                    return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def _dispatch(self, message: dict) -> None:
        if isinstance(message, list):
            for item in message:
                await self._dispatch(item)
            return
        if not isinstance(message, dict):
            return
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
        if self.http is None:
            raise RuntimeError(f"MCP server {self.name} is closed")
        headers = {"MCP-Protocol-Version": getattr(self, "version", "2025-03-26")}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.bearer():
            headers["Authorization"] = "Bearer " + self.bearer()
        async with self.http.stream("POST", self.config["url"], json=message, headers=headers) as response:
            if response.status_code in (401, 403):
                # Keep the status and WWW-Authenticate challenge: discovery and
                # sign-in need the advertised resource metadata URL and scope.
                raise McpAuthorizationRequired(
                    response.status_code,
                    parse_www_authenticate(response.headers.get("www-authenticate")))
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
                    reply = None
                    for item in event if isinstance(event, list) else [event]:
                        if not isinstance(item, dict):
                            continue
                        if item.get("id") == message["id"] and "method" not in item:
                            reply = item
                        else:
                            await self._dispatch(item)
                    if reply is not None:
                        return reply
            raise ConnectionError("MCP stream disconnected before a response")

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
        """Close the transport; resolves once nothing of this server is open, including a
        transport that was still connecting: an in-flight initialize fails against the
        closed transport and the connect aborts.

        Concurrent callers share one teardown task, so a second close waits for the
        cleanup the first started rather than returning once the fields look empty.
        """

        self.shutdown.set()
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._teardown_transports())
        # Shield the shared task: a cancelled caller must not abort cleanup that a
        # concurrent close (or the manager) is still waiting for.
        await asyncio.shield(self._close_task)

    async def _teardown_transports(self) -> None:
        """Stop the pump and close the HTTP client or stdio process. Idempotent: a connect
        that fails or is cancelled tears the same transports down after a concurrent close."""

        await self._stop_pump()
        if self.http:
            http, self.http = self.http, None
            try:
                await http.aclose()
            except Exception:  # noqa: BLE001 - closing never fails the shutdown
                pass
        if self.process:
            process, self.process = self.process, None
            if process.returncode is None:
                if sys.platform == "win32":
                    killer = await asyncio.create_subprocess_exec("taskkill", "/F", "/T", "/PID", str(process.pid),
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    await killer.wait()
                else:
                    try:
                        os.killpg(process.pid, process_signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
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
        self.on_state_changed = None  # callback(server, state, error=None): connecting/connected/failed/disabled/login-required
        self.lazy_servers = set()
        self.log_path = config.agent_dir / "mcp.log"
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._reconnect_locks: Dict[str, asyncio.Lock] = {}
        self.extension_servers: Dict[str, dict] = {}
        self.extension_server_owners: Dict[str, str] = {}
        # Lifecycle: every reload opens a new generation. A close invalidates the
        # current generation before waiting, so a connect that finishes late cannot
        # register a client into the generation the caller just shut down.
        # Created on first async use: asyncio.Lock() binds to the running loop, and a
        # manager may be constructed before the loop exists (Python 3.9).
        self._state_lock: Optional[asyncio.Lock] = None
        self._reload_lock: Optional[asyncio.Lock] = None
        self._generation = 0
        self._active = True
        # id(client) -> (client, task) for connects that have not settled yet.
        self._opening: Dict[int, tuple] = {}
        # Notification-triggered refreshes, cancelled and awaited by close().
        self._refresh_tasks: set = set()
        # Close tasks still running; a concurrent close waits for these too.
        self._cleanup_tasks: set = set()
        # OAuth: servers that failed for lack of a usable grant stay loginable, and
        # refreshes share one lock per credential so a rotated refresh token is not
        # spent twice. Logout bumps the epoch to discard late refresh results.
        self.login_servers: Dict[str, dict] = {}
        self.login_challenges: Dict[str, dict] = {}
        self._oauth_locks: Dict[str, asyncio.Lock] = {}
        self._oauth_tasks: set = set()
        self._oauth_epoch: Dict[str, int] = {}
        # In-flight ``oauth_login`` tasks, keyed by server: close() cancels them so
        # a session shutdown does not leave a sign-in (and its callback server) running.
        self._login_tasks: Dict[str, asyncio.Task] = {}

    def _ensure_locks(self) -> None:
        """Create the lifecycle locks inside the running loop (Python 3.9 binding)."""

        if self._state_lock is None:
            self._state_lock = asyncio.Lock()
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()

    def _log_message(self, server: str, params: dict) -> None:
        """Append one ``notifications/message`` to mcp.log; continuation lines are indented."""

        level = params.get("level", "info") if isinstance(params, dict) else "info"
        data = params.get("data") if isinstance(params, dict) else params
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
        text_lines = text.splitlines()
        first = text_lines[0] if text_lines else ""
        lines = "\n".join(("    " + line) for line in text_lines[1:])
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{timestamp} [{level}] {server}: {first}\n")
                if lines:
                    handle.write(lines + "\n")
        except OSError:
            pass

    async def reload(self, wait: bool = True) -> None:
        """Re-read the configuration and connect every server.

        With ``wait`` the new connections settle before this returns (tests,
        ``/mcp reconnect``). Otherwise connects run in the background: a slow or
        stalled server cannot hold startup, ``/reload`` or extension registration
        hostage, and tools arrive through ``on_tools_changed`` as each server lands.
        Late connections cannot register into a closed generation.
        """

        self._ensure_locks()
        async with self._reload_lock:
            await self._close_all()
            async with self._state_lock:
                self._generation += 1
                generation = self._generation
                self._active = True
                self.errors, self.tools = {}, []
                self._refresh_locks = {}
            await self._reload_servers(generation, wait)

    async def _reload_servers(self, generation: int, wait: bool = True) -> None:
        servers = {}
        paths = [self.config.agent_dir / "mcp.json"]
        if self.config.project_trusted:
            paths.append(self.config.cwd / ".pi/mcp.json")
        for path in paths:
            entries = read_json(path).get("mcpServers", {})
            if not isinstance(entries, dict):
                self.errors[str(path)] = "mcpServers must be an object"
                await self._notify_state(str(path), "failed", "mcpServers must be an object")
                continue
            for name, value in entries.items():
                if not isinstance(value, dict):
                    self.errors[name] = "Server configuration must be an object"
                    await self._notify_state(name, "failed", "Server configuration must be an object")
                    continue
                servers[name] = {**servers.get(name, {}), **value}
        # File configuration has precedence over ephemeral extension registrations.
        servers = {**self.extension_servers, **servers}
        normalized = set()
        connects: List[asyncio.Task] = []
        client: Optional[McpClient] = None
        for name, settings in servers.items():
            client = None
            try:
                namespace = re.sub(r"[^A-Za-z0-9_]", "_", name)
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or namespace in normalized:
                    raise ValueError("Invalid or colliding server name")
                normalized.add(namespace)
                if not settings.get("enabled", True):
                    self.errors[name] = "disabled"
                    await self._notify_state(name, "disabled")
                    continue
                if ("command" in settings) == ("url" in settings):
                    raise ValueError("Specify either command or url")
                if settings.get("type") not in (None, "stdio", "http", "streamable-http"):
                    raise ValueError("Unsupported MCP transport")
                timeout = settings.get("timeout", 60)
                if settings.get("exposure", "direct") not in ("direct", "deferred", "codemode"):
                    raise ValueError("exposure must be direct, deferred, or codemode")
                if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
                    raise ValueError("timeout must be between 0 and 86400 seconds")
                if "command" in settings and (not isinstance(settings["command"], str) or not isinstance(settings.get("args", []), list) or not all(isinstance(arg, str) for arg in settings.get("args", []))):
                    raise ValueError("command must be an executable and args a string list")
                for field in ("env", "headers"):
                    values = settings.get(field, {})
                    if (not isinstance(values, dict)
                            or not all(isinstance(key, str) and isinstance(value, str)
                                       for key, value in values.items())):
                        raise ValueError(f"{field} must map strings to strings")
                if "command" in settings:
                    # cmd.exe treats newlines as command boundaries even inside
                    # the escaping used for Windows batch shims.
                    if any("\r" in value or "\n" in value
                           for value in [settings["command"], *settings.get("args", [])]):
                        raise ValueError("stdio command and args must not contain CR or LF")
                if "url" in settings:
                    validate_url(settings["url"])
                oauth = settings.get("oauth")
                if oauth is not None:
                    if not isinstance(oauth, dict):
                        raise ValueError("oauth must be an object")
                    # Static endpoints remain optional: without them the endpoints are
                    # discovered from the server's 401 challenge and metadata documents.
                    validate_oauth_settings(oauth)
                client = McpClient(name, settings, self.config.cwd)
                if client.lazy and "url" not in settings:
                    raise ValueError("lazy tool discovery requires an HTTP server")
                # Connect in a tracked task. The registration into ``_opening``
                # happens here under the state lock, so a concurrent close sees
                # every connect and waits for it, even one that never ran yet.
                task = asyncio.ensure_future(
                    self._connect_and_register(client, namespace, generation, name))
                async with self._state_lock:
                    self._opening[id(client)] = (client, task)
                connects.append(task)
            except asyncio.CancelledError:
                # A cancelled reload must not leave the half-opened transport running.
                if client is not None:
                    await client.close()
                raise
            except Exception as exc:  # noqa: BLE001 - one invalid server must not abort reload
                if self._active and generation == self._generation:
                    self.errors[name] = str(exc) or type(exc).__name__
                    await self._notify_state(name, "failed", self.errors[name])
                if client is not None:
                    await client.close()
        if wait and connects:
            await asyncio.gather(*connects, return_exceptions=True)

    async def _connect_and_register(self, client: McpClient, namespace: str, generation: int,
                                    register_name: str) -> None:
        """Connect one validated server and register its tools, for both the waiting
        and the background reload. Failures land in ``errors`` and are broadcast."""

        name = register_name
        await self._notify_state(name, "connecting")
        try:
            await self._open_client(client, namespace, generation, register_name=name)
            async with self._state_lock:
                if not self._active or generation != self._generation:
                    raise RuntimeError(f"MCP server {name} connection cancelled")
                if client.lazy:
                    self.lazy_servers.add(name)
                else:
                    self._register_server_tools(namespace, client)
                if client.resources_supported:
                    self._register_resource_tools()
            if client.tools_changed:
                # A list_changed notification raced the connect; list again now that
                # the client is registered.
                await self.ensure_tools(name, generation)
            if self.on_tools_changed is not None:
                await _maybe_call(self.on_tools_changed, name)
            await self._notify_state(name, "connected")
        except asyncio.CancelledError:
            # A cancelled reload must not leave the half-opened transport running.
            await client.close()
            raise
        except Exception as exc:  # noqa: BLE001 - one failed server must not abort reload
            error = str(exc) or type(exc).__name__
            if self._active and generation == self._generation:
                self.errors[name] = error
                state = "login-required" if isinstance(exc, McpLoginRequired) else "failed"
                await self._notify_state(name, state, error)
            await client.close()
            async with self._state_lock:
                if self.clients.get(name) is client:
                    self.clients.pop(name, None)
                    self.tools = [tool for tool in self.tools
                                  if not getattr(tool, "_mcp_server", None) == name]

    async def _notify_state(self, server: str, state: str, error: Optional[str] = None) -> None:
        """Broadcast one server lifecycle transition to the host."""

        callback = self.on_state_changed
        if callback is None:
            return
        try:
            await _maybe_call(callback, server, state, error)
        except Exception:  # noqa: BLE001 - a broken listener must not break connects
            pass

    def server_states(self) -> Dict[str, str]:
        """Per-server status for ``/mcp``: connecting, connected/lazy tools, sign-in
        needed, or the configured error (disabled, invalid, failed connect)."""

        states: Dict[str, str] = {}
        for client, _task in self._opening.values():
            states[client.name] = "connecting"
        for name, client in self.clients.items():
            states[name] = "lazy" if name in self.lazy_servers else f"connected, {len(client.tools)} tools"
        for name in self.login_servers:
            states[name] = "requires sign-in (run /mcp login <server>)"
        for name, error in self.errors.items():
            states.setdefault(name, error)
        return states

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

    async def _open_client(self, client: McpClient, namespace: str, generation: int,
                           register_name: Optional[str] = None) -> None:
        """Connect a client registered as opening for ``generation``.

        The registration and the move into ``clients`` happen under the state lock,
        so close() sees every connect either as opening or as registered and never
        misses one that finished in between. A client whose generation was closed
        while connecting is not registered; the caller closes it.
        """

        self._ensure_locks()
        async with self._state_lock:
            if not self._active or generation != self._generation:
                raise RuntimeError(f"MCP server {client.name} connection cancelled")
            self._opening[id(client)] = (client, asyncio.current_task())
        try:
            await self._connect_client(client, namespace)
            async with self._state_lock:
                if not self._active or generation != self._generation:
                    raise RuntimeError(f"MCP server {client.name} connection cancelled")
                if register_name is not None:
                    self.clients[register_name] = client
        finally:
            async with self._state_lock:
                self._opening.pop(id(client), None)

    @staticmethod
    def _uses_oauth(settings: dict) -> bool:
        """HTTP servers authenticate with OAuth unless the config supplies Authorization."""

        if "url" not in settings:
            return False
        headers = settings.get("headers") or {}
        return not any(str(name).lower() == "authorization" for name in headers)

    @staticmethod
    def _oauth_key(server: str, settings: dict) -> str:
        return oauth_key(re.sub(r"[^A-Za-z0-9_]", "_", server), settings["url"])

    def _remember_login(self, server: str, settings: dict, challenge: Optional[dict] = None) -> None:
        # The failed client is removed from ``clients``, so sign-in needs the config
        # (and the challenge's scope/resource metadata) kept separately.
        self.login_servers[server] = dict(settings)
        if challenge:
            self.login_challenges[server] = dict(challenge)

    @staticmethod
    def _login_message(server: str) -> str:
        return f'MCP server "{server}" requires sign-in. Run /mcp login {server}'

    async def _connect_client(self, client: McpClient, namespace: str) -> None:
        """Connect with OAuth credentials when configured; never auto-launches a browser."""
        client.credential = None
        client.on_log_message = self._log_message
        client.on_tools_changed = self._schedule_tools_changed
        auth_header = ""
        if self._uses_oauth(client.config):
            key = self._oauth_key(client.name, client.config)
            try:
                record = await self._initial_credential(client.name, client.config)
            except McpLoginRequired:
                record = self.oauth_store.get(key)
            except (McpOAuthError, httpx.HTTPError, ValueError, OSError):
                # A refresh failure must not block the connection: send the stored
                # token and let the server's 401 decide whether sign-in is needed.
                record = self.oauth_store.get(key)
            if record.get("access"):
                client.credential = record
                auth_header = record["access"]
        try:
            await client.connect(auth_header)
        except McpAuthorizationRequired as exc:
            client.needs_login = True
            client.challenge = exc.challenge
            # Discover endpoints from the challenge now, so sign-in does not have to
            # and `/mcp` can report a loginable server. No browser is started here.
            key = self._oauth_key(client.name, client.config)
            record = self.oauth_store.get(key)
            try:
                async with httpx.AsyncClient(timeout=OAUTH_REQUEST_TIMEOUT_SECONDS) as http:
                    discovery = await self._discover_oauth(client.config, record, exc.challenge, http)
                self.oauth_store.set(key, {**record, "discovery": discovery})
            except (McpOAuthError, httpx.HTTPError, ValueError, OSError):
                pass
            self._remember_login(client.name, client.config, exc.challenge)
            raise McpLoginRequired(self._login_message(client.name)) from None

    async def _initial_credential(self, server: str, settings: dict) -> dict:
        record = self.oauth_store.get(self._oauth_key(server, settings))
        expires = record.get("expires")
        if record.get("access") and (expires is None or expires > time.time() + 30):
            return record
        if record.get("refresh"):
            return await self._refresh_tracked(server, settings, record)
        return record

    def _oauth_task_done(self, task: asyncio.Task) -> None:
        self._oauth_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _refresh_tracked(self, server: str, settings: dict, record: dict,
                               challenge: Optional[dict] = None) -> dict:
        """Run a refresh as a task so close() can wait for a token rotation to be saved."""

        task = asyncio.ensure_future(self._refresh_oauth(server, settings, record, challenge))
        self._oauth_tasks.add(task)
        task.add_done_callback(self._oauth_task_done)
        return await asyncio.shield(task)

    async def _refresh_oauth(self, server: str, settings: dict, record: dict,
                             challenge: Optional[dict] = None) -> dict:
        oauth = settings.get("oauth") or {}
        key = self._oauth_key(server, settings)
        lock = self._oauth_locks.setdefault(key, asyncio.Lock())
        epoch = self._oauth_epoch.get(key, 0)
        async with lock:
            latest = self.oauth_store.get(key)
            if epoch != self._oauth_epoch.get(key, 0):
                raise McpLoginRequired(self._login_message(server))
            if latest.get("access") and latest.get("access") != record.get("access"):
                # Another caller (or process) refreshed while this one waited.
                return latest
            if not latest.get("refresh"):
                raise McpLoginRequired(self._login_message(server))
            client_info = latest.get("client") or static_client(oauth)
            if not client_info:
                raise McpLoginRequired(self._login_message(server))
            async with httpx.AsyncClient(timeout=OAUTH_REQUEST_TIMEOUT_SECONDS) as http:
                discovery = await self._discover_oauth(settings, latest, challenge, http)
                metadata = discovery.get("authorization_server_metadata")
                resource = select_resource(settings["url"], discovery.get("resource_metadata"))
                tokens = await refresh_authorization(
                    http, discovery["authorization_server_url"], client_info, latest["refresh"],
                    metadata, resource, token_url=oauth_settings(oauth)["token_url"])
            # Once the server answered, persist the rotation even if close() runs now.
            task = asyncio.current_task()
            if task is not None:
                task.oauth_saving = True
            updated = record_from_tokens(
                {**latest, "client": client_info, "discovery": discovery}, tokens, latest.get("scope"))
            await self._persist_oauth_record(key, updated, epoch, server)
            return updated

    async def _persist_oauth_record(self, key: str, record: dict, epoch: int, server: str) -> None:
        """Save a rotated grant unless a logout (or another refresh) invalidated it."""

        if epoch != self._oauth_epoch.get(key, 0) or not self.oauth_store.get(key).get("refresh"):
            raise McpLoginRequired(self._login_message(server))
        self.oauth_store.set(key, record)

    async def _discover_oauth(self, settings: dict, record: dict, challenge: Optional[dict],
                              http: httpx.AsyncClient) -> dict:
        oauth = oauth_settings(settings.get("oauth") or {})
        cached = record.get("discovery") or {}
        if oauth["authorize_url"] and oauth["token_url"]:
            server = urlsplit(settings["url"])
            issuer = cached.get("authorization_server_url") or f"{server.scheme}://{server.netloc}/"
            metadata = cached.get("authorization_server_metadata") or {
                "issuer": issuer,
                "authorization_endpoint": oauth["authorize_url"],
                "token_endpoint": oauth["token_url"],
                "registration_endpoint": None,
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
            }
            return {"authorization_server_url": issuer,
                    "authorization_server_metadata": metadata,
                    "resource_metadata": cached.get("resource_metadata")}
        if not oauth["auth_server_metadata_url"] and cached.get("authorization_server_url"):
            metadata = cached.get("authorization_server_metadata")
            if metadata is None:
                metadata = await discover_authorization_server_metadata(
                    cached["authorization_server_url"], http)
            return {**cached, "authorization_server_metadata": metadata}
        return await discover_oauth_server_info(
            settings["url"], http,
            (challenge or {}).get("resource_metadata"),
            oauth["auth_server_metadata_url"])

    def _schedule_tools_changed(self, server: str) -> None:
        """Track a notification-triggered refresh so close() can cancel and await it."""

        try:
            task = asyncio.get_running_loop().create_task(self._handle_tools_changed(server))
        except RuntimeError:
            return
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)

    async def _handle_tools_changed(self, server: str) -> None:
        """Refresh one server's tools after notifications/tools/list_changed. A server that
        is not registered yet is still connecting; reload lists its tools anyway."""

        generation = self._generation
        client = self.clients.get(server)
        if client is None or not self._active:
            return
        try:
            await self.ensure_tools(server, generation)
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
            if self._active and generation == self._generation:
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
        tool.exposure = self.clients[server].config.get("exposure", "direct")
        tool.declared = tool.exposure == "direct"
        tool._mcp_server = server
        return tool

    def _register_resource_tools(self) -> None:
        """The resource tools Codex and opencode use: list, list templates, read. With a
        ``server`` argument one page is listed and ``cursor`` continues it; without, every
        page of every server, with per-server errors collected. Re-entrant: it is called
        after every registration, so previous copies are replaced, not duplicated."""

        self.tools = [tool for tool in self.tools if not getattr(tool, "_mcp_resource_tool", False)]
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
        self.tools[-1]._mcp_resource_tool = True
        self.tools.append(AgentTool(name="list_mcp_resource_templates",
            description="Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to "
                        "share data that takes parameters and provides context to language models.",
            parameters=list_params, execute=lambda *args, **kwargs: list_call(*args, key="resourceTemplates", method="resources/templates/list", **kwargs)))
        self.tools[-1]._mcp_resource_tool = True
        self.tools.append(AgentTool(name="read_mcp_resource",
            description="Read a specific resource from an MCP server given the server name and resource URI.",
            parameters={"type": "object",
                        "properties": {"server": {"type": "string"}, "uri": {"type": "string"}},
                        "required": ["server", "uri"]},
            execute=read_resource))
        self.tools[-1]._mcp_resource_tool = True

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
        used_token = client.bearer()
        try:
            return await client.request(method, params)
        except McpAuthorizationRequired as exc:
            if not self._uses_oauth(client.config):
                raise
            key = self._oauth_key(server, client.config)
            current = self.oauth_store.get(key)
            if not (used_token and current.get("access") and current["access"] != used_token):
                # The rejected token is still current: refresh once. If another
                # caller already refreshed, the per-credential lock returns its
                # result instead of spending the rotated refresh token twice.
                try:
                    current = await self._refresh_tracked(
                        server, client.config, current, exc.challenge)
                except McpLoginRequired:
                    client.needs_login = True
                    client.challenge = exc.challenge
                    self._remember_login(server, client.config, exc.challenge)
                    raise
            client.credential = current
            if signal:
                signal.throw_if_aborted()
            return await client.request(method, params)
        except (RuntimeError, httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            message = str(exc)
            transient = message == "MCP server disconnected" or re.fullmatch(r"MCP HTTP [0-9]{3}", message) is not None or isinstance(exc, httpx.HTTPError) \
                or isinstance(exc, (asyncio.TimeoutError, OSError))
            if not transient:
                raise
            await self.reconnect(server, client)
            if signal:
                signal.throw_if_aborted()
            return await self.clients[server].request(method, params)

    async def reconnect(self, server: str, client: Optional[McpClient] = None) -> None:
        self._ensure_locks()
        if client is None:
            client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        lock = self._reconnect_locks.setdefault(server, asyncio.Lock())
        async with lock:
            # A concurrent caller may already have replaced the failed connection.
            if self.clients.get(server) is not client:
                return
            generation = self._generation
            namespace = re.sub(r"[^A-Za-z0-9_]", "_", server)
            replacement = McpClient(server, client.config, client.cwd)
            try:
                await self._open_client(replacement, namespace, generation)
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
            async with self._state_lock:
                current = (self._active and generation == self._generation
                           and self.clients.get(server) is client)
                if current:
                    self.clients[server] = replacement
            if not current:
                # The manager closed (or a reload replaced the client) while the
                # replacement was connecting: release it instead of resurrecting it.
                await replacement.close()
                raise RuntimeError(f"MCP server {server} is closed")
            await self.ensure_tools(server)
    

    async def ensure_tools(self, server: str, generation: Optional[int] = None) -> None:
        """Discover tools for a lazily-connected server and refresh changed tool sets."""

        self._ensure_locks()
        if generation is None:
            generation = self._generation
        client = self.clients.get(server)
        if client is None:
            raise ValueError(f"MCP server not connected: {server}")
        lock = self._refresh_locks.setdefault(server, asyncio.Lock())
        async with lock:
            namespace = re.sub(r"[^A-Za-z0-9_]", "_", server)
            before = [(tool.name, tool.description, tool.parameters) for tool in self.tools]
            if client.tools_changed or not client.tools:
                await client.list_tools()
                client.tools_changed = False
            # A close or reload may have run while tools were being listed. The
            # closed generation must not publish declarations afterwards.
            async with self._state_lock:
                if (not self._active or generation != self._generation
                        or self.clients.get(server) is not client):
                    return
                self.tools = [tool for tool in self.tools if getattr(tool, "_mcp_server", None) != server]
                self._register_server_tools(namespace, client)
                self.lazy_servers.discard(server)
                after = [(tool.name, tool.description, tool.parameters) for tool in self.tools]
            if self.on_tools_changed is not None and after != before:
                await _maybe_call(self.on_tools_changed, server)

    async def refresh_changed(self) -> bool:
        """Re-list tools for servers that reported notifications/tools/list_changed."""
        generation = self._generation
        changed = [name for name, client in self.clients.items() if client.tools_changed]
        for name in changed:
            await self.ensure_tools(name, generation)
        return bool(changed)

    async def oauth_login(self, server: str, signal=None) -> None:
        """Run the browser authorization flow, save the grant and connect the server.

        Works after the first 401 removed the failed client: the configuration and
        challenge are kept in ``login_servers``/``login_challenges``. ``signal``
        aborts the flow (raising :class:`AbortError`), including an in-flight
        request; a session shutdown cancels the task through ``close()`` instead.
        The task is tracked per server, so both paths close the callback server
        and save no credentials.
        """

        task = asyncio.current_task()
        self._login_tasks[server] = task
        try:
            if signal is None:
                await self._oauth_login(server, None)
                return
            inner = asyncio.ensure_future(self._oauth_login(server, signal))
            abort = asyncio.ensure_future(signal.wait())
            try:
                await asyncio.wait({inner, abort}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                inner.cancel()
                abort.cancel()
                await asyncio.gather(inner, abort, return_exceptions=True)
            signal.throw_if_aborted()
            await asyncio.gather(inner)
        finally:
            if self._login_tasks.get(server) is task:
                self._login_tasks.pop(server, None)

    async def _oauth_login(self, server: str, signal) -> None:
        client = self.clients.get(server)
        settings = client.config if client is not None else self.login_servers.get(server)
        if settings is None or "url" not in settings:
            raise ValueError(f"MCP server not connected: {server}")
        if signal is not None:
            signal.throw_if_aborted()
        oauth = settings.get("oauth") or {}
        values = validate_oauth_settings(oauth)
        key = self._oauth_key(server, settings)
        # A refresh already in flight must not overwrite the new sign-in.
        self._oauth_epoch[key] = self._oauth_epoch.get(key, 0) + 1
        challenge = self.login_challenges.get(server, {})
        callback = callback_settings(oauth)
        state = secrets.token_hex(16)
        callback_server = await start_oauth_callback_server(
            server, callback["listen_host"], callback["port"] or 0, callback["path"],
            state=state, timeout_ms=int(values["timeout"] * 1000))
        try:
            redirect_url = callback["fixed_redirect_url"] or callback_server.redirect_uri
            async with httpx.AsyncClient(timeout=OAUTH_REQUEST_TIMEOUT_SECONDS) as http:
                record = self.oauth_store.get(key)
                record, authorization_url = await self._prepare_login(
                    server, settings, record, redirect_url, state, challenge, http)
                if signal is not None:
                    signal.throw_if_aborted()
                # Keep the registered client and PKCE verifier even if the user never
                # finishes; a later sign-in then reuses the registration.
                self.oauth_store.set(key, record)
                self.notify(f"MCP authorization required for {settings['url']}\nOpen: {authorization_url}")
                params = await self._wait_callback_or_abort(callback_server, signal)
                if not params:
                    raise ValueError("MCP sign-in cancelled")
                if signal is not None:
                    signal.throw_if_aborted()
                record = await self._complete_login(server, settings, record, params, http)
                self.oauth_store.set(key, record)
        finally:
            await callback_server.close()
        self.login_challenges.pop(server, None)
        self.errors.pop(server, None)
        await self._connect_after_login(server, settings)

    @staticmethod
    async def _wait_callback_or_abort(callback_server, signal):
        """Wait for the browser callback; an abort fires :class:`AbortError` while
        the callback server stays open for the caller's cleanup."""

        if signal is None:
            return await callback_server.wait()
        waiter = asyncio.ensure_future(callback_server.wait())
        abort = asyncio.ensure_future(signal.wait())
        try:
            await asyncio.wait({waiter, abort}, return_when=asyncio.FIRST_COMPLETED)
            if abort.done() and not waiter.done():
                raise AbortError("MCP sign-in cancelled")
            return waiter.result()
        finally:
            waiter.cancel()
            abort.cancel()
            await asyncio.gather(waiter, abort, return_exceptions=True)

    async def _prepare_login(self, server: str, settings: dict, record: dict, redirect_url: str,
                             state: str, challenge: dict, http: httpx.AsyncClient):
        oauth = settings.get("oauth") or {}
        values = oauth_settings(oauth)
        discovery = await self._discover_oauth(settings, record, challenge, http)
        metadata = discovery.get("authorization_server_metadata")
        if metadata is None:
            raise McpOAuthError(
                "discovery_failed",
                f"Could not discover authorization server metadata for {settings['url']}")
        resource = select_resource(settings["url"], discovery.get("resource_metadata"))
        scope = (merge_scopes(values["scope"], challenge.get("scope"))
                 or " ".join((discovery.get("resource_metadata") or {}).get("scopes_supported") or [])
                 or None)
        client_info = record.get("client") or static_client(oauth)
        if not client_info:
            client_info = await register_client(
                http, discovery["authorization_server_url"],
                client_metadata(oauth, redirect_url), metadata, scope)
        authorization_url, verifier = start_authorization(
            discovery["authorization_server_url"], client_info, redirect_url, metadata,
            scope, state, resource, authorize_url=values["authorize_url"])
        updated = {**record, "client": client_info, "discovery": discovery,
                   "code_verifier": verifier, "oauth_state": state,
                   "redirect_uri": redirect_url, "scope": scope}
        return updated, authorization_url

    async def _complete_login(self, server: str, settings: dict, record: dict,
                              params: dict, http: httpx.AsyncClient) -> dict:
        code = params.get("code")
        if not code:
            raise ValueError("OAuth callback did not include an authorization code")
        discovery = record.get("discovery") or {}
        metadata = discovery.get("authorization_server_metadata")
        iss = params.get("iss")
        # RFC 9207: never send a code from another authorization server to this one.
        if metadata and (iss is not None or metadata.get("authorization_response_iss_parameter_supported")):
            expected = metadata["issuer"].rstrip("/")
            if (iss or "").rstrip("/") != expected:
                raise OAuthIssuerMismatchError(expected, iss)
        oauth = settings.get("oauth") or {}
        values = oauth_settings(oauth)
        client_info = record.get("client") or static_client(oauth)
        if not client_info:
            raise McpOAuthError("invalid_client", "OAuth client information is missing during code exchange")
        resource = select_resource(settings["url"], discovery.get("resource_metadata"))
        tokens = await exchange_authorization_code(
            http, discovery["authorization_server_url"], client_info, code,
            record["code_verifier"], record["redirect_uri"], metadata, resource,
            token_url=values["token_url"])
        updated = record_from_tokens(record, tokens, record.get("scope"))
        updated.pop("code_verifier", None)
        updated.pop("oauth_state", None)
        return updated

    async def _connect_after_login(self, server: str, settings: dict) -> None:
        self._ensure_locks()
        async with self._state_lock:
            if not self._active:
                return
            generation = self._generation
            existing = self.clients.pop(server, None)
            self.lazy_servers.discard(server)
            self.tools = [tool for tool in self.tools
                          if not getattr(tool, "_mcp_server", None) == server]
        if existing is not None:
            await existing.close()
        namespace = re.sub(r"[^A-Za-z0-9_]", "_", server)
        client = McpClient(server, settings, self.config.cwd)
        await self._open_client(client, namespace, generation, register_name=server)
        async with self._state_lock:
            if not self._active or generation != self._generation:
                return
            if client.lazy:
                self.lazy_servers.add(server)
            else:
                self._register_server_tools(namespace, client)
        if client.tools_changed:
            await self.ensure_tools(server, generation)
        if self.on_tools_changed is not None:
            await _maybe_call(self.on_tools_changed, server)

    async def oauth_logout(self, server: str) -> None:
        client = self.clients.get(server)
        settings = client.config if client is not None else self.login_servers.get(server)
        if settings is None or "url" not in settings:
            raise ValueError(f"MCP server not connected: {server}")
        key = self._oauth_key(server, settings)
        # A refresh that already got a response must not write the grant back.
        self._oauth_epoch[key] = self._oauth_epoch.get(key, 0) + 1
        self.oauth_store.remove(key)
        self.login_challenges.pop(server, None)
        self._remember_login(server, settings)
        async with self._state_lock:
            removed = self.clients.pop(server, None)
            self.lazy_servers.discard(server)
            self.tools = [tool for tool in self.tools
                          if not getattr(tool, "_mcp_server", None) == server]
        if removed is not None:
            await removed.close()
        if self.on_tools_changed is not None:
            await _maybe_call(self.on_tools_changed, server)

    async def close(self):
        """Close the current generation and wait until every connect, transport and
        notification refresh of that generation has settled.

        Concurrent callers all wait for the same cleanup tasks; the state lock is
        only held while invalidating the generation and taking the snapshot, never
        while waiting for a connect that may need that lock to finish.
        """

        self._ensure_locks()
        await self._close_all()

    async def _close_all(self) -> None:
        async with self._state_lock:
            # Invalidate the generation before any await: a connect that completes
            # from here on cannot register itself into the generation being closed.
            self._generation += 1
            self._active = False
            clients = list(self.clients.values())
            self.clients.clear()
            opening = list(self._opening.values())
            self._opening.clear()
            self.lazy_servers.clear()
            self.login_servers.clear()
            self.login_challenges.clear()
            refresh_tasks = list(self._refresh_tasks)
            self._refresh_tasks.clear()
            oauth_tasks = list(self._oauth_tasks)
            login_tasks = list(self._login_tasks.values())
            self._login_tasks.clear()
            tasks = set(self._cleanup_tasks)
            for client in clients:
                tasks.add(self._track_cleanup(self._safe_close(client)))
            for client, _ in opening:
                tasks.add(self._track_cleanup(self._safe_close(client)))
            current = asyncio.current_task()
            for task in refresh_tasks:
                if task is not current:
                    task.cancel()
                    tasks.add(self._track_cleanup(self._settle(task)))
            for _, task in opening:
                if task is not None and task is not current:
                    tasks.add(self._track_cleanup(self._settle(task)))
            for task in oauth_tasks:
                if task is current:
                    continue
                # Wait for in-flight refreshes instead of cancelling them: cancelling
                # between "the server rotated the grant" and "the client saved it"
                # loses the rotated refresh token and forces a new sign-in. The wait
                # is bounded by the per-request timeout, so shutdown cannot hang on a
                # stalled network; anything still running past the deadline is cancelled.
                tasks.add(self._track_cleanup(self._settle_refresh(task)))
            for task in login_tasks:
                if task is not current:
                    # A sign-in holds no saved grant yet: abort it instead of waiting.
                    task.cancel()
                    tasks.add(self._track_cleanup(self._settle(task)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _track_cleanup(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        return task

    async def _safe_close(self, client: McpClient) -> None:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 - one failed client must not block the rest
            pass

    async def _settle(self, task: asyncio.Task) -> None:
        try:
            await task
        except BaseException:  # noqa: BLE001 - cleanup waits, it does not re-raise
            pass

    async def _settle_refresh(self, task: asyncio.Task) -> None:
        """Wait for one in-flight refresh so a rotated grant is saved before shutdown
        moves on; a task still running past the deadline is cancelled."""

        try:
            await asyncio.wait_for(asyncio.shield(task), OAUTH_REQUEST_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - timeout or failure: cancel, never re-raise
            task.cancel()
        await self._settle(task)


async def _maybe_call(fn, *args):
    result = fn(*args)
    if inspect.isawaitable(result):
        await result
