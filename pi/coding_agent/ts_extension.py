"""Node bridge that runs TypeScript/JavaScript pi extensions from Python.

The TS extension ecosystem loads extensions with ``setup(api)`` where ``api``
is an :class:`ExtensionAPI`. This module spawns one ``node`` process per
extension with a small loader (``assets/ts_extension_loader.mjs``) that
imports the extension and exposes its registrations over stdio JSON lines:

- ``registerTool`` / ``registerCommand`` / ``registerProvider`` /
  ``registerKeybinding`` become native registrations on the shared
  :class:`~pi.coding_agent.extensions.ExtensionAPI`, so the manager's
  existing tools/commands/events machinery drives them unchanged.
- ``on(event, handler)`` handlers run in Node; events cross the bridge and
  results (tool_call blocks, input transforms) are mapped back.
- ``pi.ui.*``, ``pi.appendEntry``, ``pi.sendMessage`` and ``pi.log`` route
  back to Python.

TypeScript is loaded through Node's type stripping, so extensions must stick
to erasable syntax (the same rule pi's own TS sources follow). Node 22.6+
with ``--experimental-strip-types`` or Node 23.6+ (stripping by default)
works; when neither is available the extension is skipped with a notice.

Not every TS API is bridged: TUI renderers, virtual models, MCP server
registration and session-management actions stay unimplemented and raise a
clear error when an extension calls them.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..agent.types import AgentTool, AgentToolResult, BeforeToolCallResult
from ..ai.types import ImageContent, TextContent

if TYPE_CHECKING:
    from .extensions import ExtensionAPI

LOADER_PATH = Path(__file__).parent / "assets" / "ts_extension_loader.mjs"
SETUP_TIMEOUT = 60.0
CALL_TIMEOUT = 600.0


@dataclass
class InputResult:
    """Result of an ``input`` handler: replace the text, or swallow it."""

    text: Optional[str] = None
    handled: bool = False


def node_command() -> Optional[List[str]]:
    """Flags that make ``node`` strip TypeScript, or None when unavailable."""

    node = shutil.which("node")
    if node is None:
        return None
    probe = subprocess.run([node, "--experimental-strip-types", "-e", "process.exit(0)"],
                           capture_output=True, timeout=15)
    if probe.returncode == 0:
        return [node, "--experimental-strip-types", "--no-warnings"]
    probe = subprocess.run([node, "-e", "process.exit(0)"], capture_output=True, timeout=15)
    if probe.returncode == 0:
        return [node, "--no-warnings"]
    return None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def result_from_json(payload: dict) -> AgentToolResult:
    """AgentToolResult JSON from the bridge back to the Python dataclass."""

    content = []
    for block in payload.get("content") or []:
        if block.get("type") == "image":
            content.append(ImageContent(data=block.get("data", ""),
                                        mime_type=block.get("mimeType", "image/png")))
        else:
            content.append(TextContent(text=str(block.get("text", ""))))
    return AgentToolResult(
        content=content or [TextContent(text="")],
        details=payload.get("details"),
        structured_content=payload.get("structuredContent"),
        is_error=bool(payload.get("isError")),
        terminate=bool(payload.get("terminate")),
    )


class TsExtensionBridge:
    """One Node process hosting one TS/JS extension; fills a shared ExtensionAPI."""

    def __init__(self, path: Path, api: "ExtensionAPI", node_argv: List[str]) -> None:
        self.path, self.api = path, api
        self.node_argv = node_argv
        self.process: Optional[asyncio.subprocess.Process] = None
        self.sequence = 0
        self.pending: Dict[int, asyncio.Future] = {}
        self.reader_task: Optional[asyncio.Task] = None
        self.updates: Dict[str, Callable] = {}
        self.tool_handlers: Dict[str, str] = {}
        self.command_handlers: Dict[str, str] = {}
        self.event_handlers: Dict[str, str] = {}
        self.keybinding_handlers: Dict[str, str] = {}
        self.closed = False

    async def start(self) -> None:
        if not LOADER_PATH.exists():
            raise RuntimeError(f"TS extension loader missing: {LOADER_PATH}")
        self.process = await asyncio.create_subprocess_exec(
            *self.node_argv, str(LOADER_PATH), str(self.path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=None,  # extension stderr goes straight to the terminal
            limit=16 * 1024 * 1024)
        self.reader_task = asyncio.create_task(self._read_loop())
        await self._request("start", {"path": str(self.path)}, SETUP_TIMEOUT)

    async def _request(self, method: str, params: dict, timeout: float = CALL_TIMEOUT) -> Any:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("TS extension bridge is closed")
        self.sequence += 1
        request_id = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        self.process.stdin.write((json.dumps({"id": request_id, "method": method, "params": params},
                                             ensure_ascii=False, default=str) + "\n").encode())
        await self.process.stdin.drain()
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"TS extension {self.path.name} timed out on {method}") from None
        finally:
            self.pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    self._fail_pending("TS extension process exited")
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("id") in self.pending:
                    future = self.pending.pop(message["id"])
                    if message.get("error") is not None:
                        future.set_exception(RuntimeError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
                    continue
                if message.get("method") is not None:
                    await self._handle_child_request(message)
                method = message.get("method")
                if method == "toolUpdate":
                    callback = self.updates.get((message.get("params") or {}).get("toolCallId", ""))
                    if callback is not None:
                        outcome = callback(result_from_json(message["params"].get("result") or {}))
                        if asyncio.iscoroutine(outcome):
                            await outcome
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError):
            self._fail_pending("TS extension bridge failed")

    def _fail_pending(self, reason: str) -> None:
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

    async def _reply(self, request: dict, result: Any = None, error: Optional[str] = None) -> None:
        if self.process is None or self.process.stdin is None:
            return
        payload = {"id": request.get("id")}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        try:
            self.process.stdin.write((json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode())
            await self.process.stdin.drain()
        except (ConnectionError, RuntimeError):
            pass

    # ---- child-initiated requests -------------------------------------------

    async def _handle_child_request(self, request: dict) -> None:
        method = request.get("method")
        params = request.get("params") or {}
        api = self.api
        try:
            if method == "registerTool":
                handler_id = params["handlerId"]
                self.tool_handlers[params["name"]] = handler_id
                api.register_tool(AgentTool(
                    name=params["name"], description=params.get("description", ""),
                    parameters=params.get("parameters") or {"type": "object", "properties": {}},
                    output_schema=params.get("outputSchema"),
                    execute=self._tool_execute(handler_id, params["name"])))
            elif method == "registerCommand":
                handler_id = params["handlerId"]
                self.command_handlers[params["name"]] = handler_id
                api.register_command(params["name"], self._command_handler(handler_id),
                                     description=params.get("description", ""))
            elif method == "registerProvider":
                api.register_provider(params["name"], params.get("config") or {})
            elif method == "registerKeybinding":
                handler_id = params["handlerId"]
                self.keybinding_handlers[params["action"]] = handler_id
                api.register_keybinding(params["action"], params.get("keys") or [],
                                        self._plain_handler(handler_id))
            elif method == "on":
                handler_id = params["handlerId"]
                self.event_handlers[handler_id] = params["event"]
                api.on(params["event"], self._event_handler(handler_id, params["event"]))
            elif method == "ui":
                result = await self._ui_call(params)
                await self._reply(request, result)
                return
            elif method == "appendEntry":
                entry_id = api.append_entry(params.get("customType", ""), params.get("data") or {})
                await self._reply(request, {"id": entry_id})
                return
            elif method == "sendMessage":
                entry_id = api.send_message(params.get("customType", ""), params.get("content", ""),
                                            bool(params.get("display", True)))
                await self._reply(request, {"id": entry_id})
                return
            elif method == "log":
                api.log(str(params.get("message", "")))
            else:
                await self._reply(request, error=f"unsupported bridge method: {method}")
                return
            await self._reply(request, {})
        except Exception as exc:  # noqa: BLE001  # reported to the child verbatim
            await self._reply(request, error=str(exc))

    async def _ui_call(self, params: dict) -> Any:
        call = params.get("call")
        ui = self.api.ui
        if call == "notify":
            await ui.notify(str(params.get("message", "")), str(params.get("level", "info")))
            return None
        if call == "select":
            return await ui.select(str(params.get("title", "")), list(params.get("options") or []))
        if call == "confirm":
            return await ui.confirm(str(params.get("message", "")))
        if call == "input":
            return await ui.input(str(params.get("message", "")), str(params.get("placeholder", "")))
        if call == "setStatus":
            await ui.set_status(str(params.get("text", "")))
            return None
        if call == "setWidget":
            await ui.set_widget(str(params.get("placement", "below")), list(params.get("lines") or []))
            return None
        if call == "setTitle":
            await ui.set_title(str(params.get("title", "")))
            return None
        raise ValueError(f"unsupported ui call: {call}")

    # ---- parent-initiated dispatches ----------------------------------------

    def _context_payload(self) -> dict:
        context = self.api.context
        return {"cwd": str(context.cwd), "hasSession": context.session is not None}

    def _tool_execute(self, handler_id: str, name: str):
        async def execute(call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
            if signal is not None:
                signal.throw_if_aborted()
            if on_update is not None:
                self.updates[call_id] = on_update
            try:
                payload = await self._request("callTool", {
                    "handlerId": handler_id, "toolCallId": call_id,
                    "params": to_jsonable(params), "context": self._context_payload()})
            finally:
                self.updates.pop(call_id, None)
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return result_from_json(payload or {})
        return execute

    def _command_handler(self, handler_id: str):
        async def handler(arguments: str):
            result = await self._request("callCommand", {
                "handlerId": handler_id, "arguments": arguments, "context": self._context_payload()})
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result["error"]))
            return result.get("result") if isinstance(result, dict) else None
        return handler

    def _plain_handler(self, handler_id: str):
        async def handler(*args):
            await self._request("callCommand", {"handlerId": handler_id, "arguments": "",
                                                "context": self._context_payload()})
        return handler

    def _event_handler(self, handler_id: str, event_name: str):
        async def handler(event, context=None):
            payload = await self._request("dispatchEvent", {
                "handlerId": handler_id, "event": to_jsonable(event),
                "context": self._context_payload()})
            if not isinstance(payload, dict) or payload.get("error"):
                return None
            result = payload.get("result")
            if event_name == "tool_call":
                if isinstance(result, dict) and result.get("block"):
                    return BeforeToolCallResult(block=True, reason=result.get("reason"))
                return None
            if event_name == "input":
                if isinstance(result, dict):
                    action = result.get("action")
                    if action == "handled":
                        return InputResult(handled=True)
                    if action == "transform":
                        return InputResult(text=result.get("text"))
                return None
            return result
        return handler

    async def dispatch(self, request: dict) -> None:
        await self._handle_child_request(request)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process is not None and self.process.stdin is not None and self.process.returncode is None:
            try:
                self.process.stdin.write(b'{"method":"shutdown"}\n')
                await self.process.stdin.drain()
            except (ConnectionError, RuntimeError):
                pass
        if self.reader_task is not None:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self.reader_task = None
        if self.process is not None and self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self._fail_pending("TS extension bridge closed")


async def load_ts_extension(path: Path, api: "ExtensionAPI") -> TsExtensionBridge:
    """Run one TS/JS extension through the Node bridge; registrations land on ``api``."""

    argv = node_command()
    if argv is None:
        raise RuntimeError("TS/JS extensions need Node.js 22.6+ on PATH")
    bridge = TsExtensionBridge(path, api, argv)
    await bridge.start()
    return bridge
