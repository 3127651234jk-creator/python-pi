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

Session naming, thinking changes and tool/state snapshots are bridged.
Synchronous getters read a snapshot refreshed before setup and each callback;
mutations return promises so callers can observe host validation failures.
Interactive components stay in Node: ``ctx.ui.custom`` renders into a view,
``setWidget``/``setHeader``/``setFooter`` factories render into the fullscreen
slots, and ``setEditorComponent`` replaces the prompt editor for input reads.
Keys and mouse events are routed back to ``handleInput``/``handleMouse`` and
``tui.requestRender`` re-renders. Not every TS API is bridged: session
navigation, theme switching, editor text helpers and some state getters stay
unimplemented. Unsupported methods exist on the
``pi`` object but raise a clear error when an extension calls them, instead
of failing as opaque TypeErrors.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..agent.types import AgentTool, AgentToolResult, BeforeToolCallResult, AfterToolCallResult
from ..ai.providers.pi_messages import wire_message
from ..ai.json_io import usage_from_json
from ..ai.types import ImageContent, TextContent
from ..ai.models import ModelError
from .config import Configuration
from .model_registry import ModelRegistry

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


@dataclass
class CustomViewFrame:
    """One render of an interactive ``ctx.ui.custom`` component."""

    lines: List[str]
    cursor: Optional[Tuple[int, int]] = None
    overlay: Optional[dict] = None
    hidden: bool = False


class CustomViewSession:
    """An open ``ctx.ui.custom`` component hosted in the Node child.

    The UI host drives the component: ``render`` returns the current lines,
    ``input`` forwards raw key data (the same bytes the TS component's
    ``handleInput`` expects) and ``mouse`` forwards normalized mouse events.
    The extension calls ``done(result)`` in Node, which sets ``done_event`` and
    resolves its promise.
    """

    def __init__(self, bridge: "TsExtensionBridge", session_id: str, options: dict) -> None:
        self.bridge, self.id, self.options = bridge, session_id, options or {}
        self.done_event = asyncio.Event()
        self.result: Any = None
        self.render_listener: Optional[Callable[[], None]] = None
        self.handle_listener: Optional[Callable[[str], None]] = None
        self.bounds: Optional[dict] = None

    def finish(self, result: Any) -> None:
        self.result = result
        self.done_event.set()

    def request_render(self) -> None:
        if self.render_listener is not None:
            self.render_listener()

    def handle_action(self, action: str) -> None:
        if self.handle_listener is not None:
            self.handle_listener(action)

    @staticmethod
    def _frame(payload: Any) -> CustomViewFrame:
        if not isinstance(payload, dict):
            return CustomViewFrame([])
        lines = [str(line) for line in payload.get("lines") or []]
        cursor = payload.get("cursor")
        point = None
        if isinstance(cursor, dict) and isinstance(cursor.get("x"), int) and isinstance(cursor.get("y"), int):
            point = (cursor["x"], cursor["y"])
        overlay = payload.get("overlay")
        return CustomViewFrame(lines, point, overlay if isinstance(overlay, dict) else None,
                               bool(payload.get("hidden")))

    def _params(self, extra: dict) -> dict:
        params = {"sessionId": self.id, **extra}
        if self.bounds is not None:
            params["bounds"] = dict(self.bounds)
        return params

    async def render(self, width: int, height: int = 24) -> CustomViewFrame:
        payload = await self.bridge.request(
            "customRender", self._params({"width": width, "height": height}))
        return self._frame(payload)

    async def input(self, data: str, width: int) -> CustomViewFrame:
        payload = await self.bridge.request(
            "customInput", self._params({"data": data, "width": width}))
        return self._frame(payload)

    async def mouse(self, event: dict, width: int) -> CustomViewFrame:
        payload = await self.bridge.request(
            "customMouse", self._params({"event": event, "width": width}))
        return self._frame(payload)


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


class CustomEditorSession:
    """An extension-owned input editor hosted in the Node child.

    ``ctx.ui.setEditorComponent(factory)`` replaces the Python prompt editor
    with the extension's component: the read loop renders it, forwards raw key
    data, and returns the text reported by the component's ``onSubmit``.
    """

    def __init__(self, bridge: "TsExtensionBridge", session_id: str) -> None:
        self.bridge, self.id = bridge, session_id
        self.submitted: Optional[str] = None
        self.action: Optional[str] = None
        self.submit_event = asyncio.Event()
        self.render_listener: Optional[Callable[[], None]] = None

    def submit(self, text: str) -> None:
        self.submitted = text
        self.submit_event.set()

    def request_action(self, action: str) -> None:
        self.action = action
        self.submit_event.set()

    def request_render(self) -> None:
        if self.render_listener is not None:
            self.render_listener()

    def reset(self) -> None:
        self.submitted = None
        self.action = None
        self.submit_event.clear()

    async def render(self, width: int, height: int = 24) -> CustomViewFrame:
        payload = await self.bridge.request(
            "editorRender", {"sessionId": self.id, "width": width, "height": height})
        return CustomViewSession._frame(payload)

    async def input(self, data: str, width: int) -> CustomViewFrame:
        payload = await self.bridge.request(
            "editorInput", {"sessionId": self.id, "data": data, "width": width})
        return CustomViewSession._frame(payload)

    async def mouse(self, event: dict, width: int) -> CustomViewFrame:
        payload = await self.bridge.request(
            "editorMouse", {"sessionId": self.id, "event": event, "width": width})
        return CustomViewSession._frame(payload)


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
        self.renderer_handlers: List[str] = []
        self.virtual_model_handlers: Dict[tuple, str] = {}
        self.custom_sessions: Dict[str, CustomViewSession] = {}
        self.custom_tasks: set = set()
        self.editor_session: Optional[CustomEditorSession] = None
        self.editor_serial = 0
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
        self.api.context.event_receivers.append(self._receive_bus_event)
        try:
            await self._request("start", {"path": str(self.path), "context": self._context_payload()}, SETUP_TIMEOUT)
        except Exception:
            await self.close()
            raise

    async def request(self, method: str, params: dict, timeout: float = CALL_TIMEOUT) -> Any:
        """Host-initiated request; used by interactive custom views."""
        return await self._request(method, params, timeout)

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
                    self._unsubscribe_bus()
                    self._fail_pending("TS extension process exited")
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                # Extensions may print JSON scalars or arrays as ordinary logs.
                # Only objects can be bridge protocol messages.
                if not isinstance(message, dict):
                    continue
                if message.get("method") is None and message.get("id") in self.pending:
                    future = self.pending.pop(message["id"])
                    if future.done():
                        continue
                    if message.get("error") is not None:
                        future.set_exception(RuntimeError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
                    continue
                method = message.get("method")
                if method in ("customDone", "customRenderRequest", "customHandle"):
                    self._handle_custom_notification(message)
                    continue
                if method in ("editorSubmit", "editorAction", "editorRenderRequest"):
                    self._handle_editor_notification(message)
                    continue
                if method == "toolUpdate":
                    callback = self.updates.get((message.get("params") or {}).get("toolCallId", ""))
                    if callback is not None:
                        outcome = callback(result_from_json(message["params"].get("result") or {}))
                        if asyncio.iscoroutine(outcome):
                            await outcome
                    continue
                if method is not None:
                    await self._handle_child_request(message)
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError):
            self._fail_pending("TS extension bridge failed")
        finally:
            self._unsubscribe_bus()

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

    async def _notify(self, method: str, params: dict) -> None:
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write((json.dumps({"method": method, "params": params},
                                                 ensure_ascii=False, default=str) + "\n").encode())
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
                if params.get("rendererId"):
                    api.register_tool_renderer(self._tool_renderer(params["rendererId"]))
            elif method == "registerCommand":
                handler_id = params["handlerId"]
                self.command_handlers[params["name"]] = handler_id
                api.register_command(params["name"], self._command_handler(handler_id),
                                     description=params.get("description", ""))
            elif method == "registerProvider":
                api.register_provider(params["name"], params.get("config") or {})
            elif method == "unregisterProvider":
                api.unregister_provider(params["name"])
            elif method == "registerFlag":
                api.register_flag(params["name"], params.get("options") or {})
            elif method == "registerToolRenderer":
                handler_id = params["handlerId"]
                self.renderer_handlers.append(handler_id)
                api.register_tool_renderer(self._tool_renderer(handler_id))
            elif method == "registerMessageRenderer":
                api.register_message_renderer(params["customType"],
                                              self._content_renderer(params["handlerId"], "message"))
            elif method == "registerEntryRenderer":
                api.register_entry_renderer(params["customType"],
                                            self._content_renderer(params["handlerId"], "entry"))
            elif method == "registerMarkdownTransformer":
                api.register_markdown_transformer(self._markdown_transformer(params["handlerId"]))
            elif method == "registerVirtualModel":
                definition = dict(params.get("definition") or {})
                key = (definition.get("provider"), definition.get("id"))
                handler_id = params["handlerId"]
                self.virtual_model_handlers[key] = handler_id
                definition["route"] = self._virtual_model_route(handler_id)
                api.register_virtual_model(definition)
            elif method == "unregisterVirtualModel":
                api.unregister_virtual_model(params["provider"], params["id"])
            elif method == "registerMcpServer":
                await api.register_mcp_server(params["name"], params.get("config") or {})
            elif method == "unregisterMcpServer":
                await api.unregister_mcp_server(params["name"])
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
            elif method == "uiCustom":
                await self._open_custom(request, params)
                return
            elif method == "uiEditor":
                await self._open_editor(request, params)
                return
            elif method == "terminalSize":
                provider = self.api.ui.size_provider
                if provider is None:
                    await self._reply(request, {"width": 80, "height": 24})
                else:
                    width, height = provider()
                    await self._reply(request, {"width": width, "height": height})
                return
            elif method == "appendEntry":
                entry_id = api.append_entry(params.get("customType", ""), params.get("data") or {})
                await self._reply(request, {"id": entry_id})
                return
            elif method == "sendMessage":
                entry_id = api.send_message(params.get("customType", ""), params.get("content", ""),
                                            bool(params.get("display", True)), params.get("details"))
                await self._reply(request, {"id": entry_id})
                return
            elif method == "log":
                api.log(str(params.get("message", "")))
            elif method == "setActiveTools":
                if api.context.agent.is_streaming:
                    raise RuntimeError("Changing tools requires an idle agent")
                self._context_payload()
                names = params["names"]
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    raise ValueError("Tool names must be a string array")
                api.context.agent.tools = [tool for name, tool in api.context.tool_catalog.items() if name in names]
                await self._reply(request, self._context_payload())
                return
            elif method == "refreshTools":
                api.refresh_tools()
                await self._reply(request, self._context_payload())
                return
            elif method == "setModel":
                if api.context.agent.is_streaming:
                    raise RuntimeError("Changing models requires an idle agent")
                registry = api.context.registry
                if registry is None:
                    registry = ModelRegistry(api.context.config or Configuration(str(api.context.cwd)))
                selected = params["model"]
                model = registry.resolve(selected["provider"], selected["id"])
                try:
                    await model.get_api_key(model.provider)
                except ModelError:
                    await self._reply(request, False)
                    return
                api.context.agent.model = model
                if api.context.session:
                    api.context.session.append_model_change(model.provider, model.id)
                await self._reply(request, True)
                return
            elif method == "setLabel":
                session = api.context.session
                if session is None:
                    raise RuntimeError("Session persistence is disabled")
                if params["entryId"] not in {entry.id for entry in session.entries}:
                    raise ValueError("Unknown entry ID")
                label = params.get("label")
                if label is not None and not isinstance(label, str):
                    raise ValueError("Label must be a string")
                session.append_custom("label", {"entryId": params["entryId"], "label": label})
            elif method == "events.emit":
                channel = params.get("channel")
                if not isinstance(channel, str):
                    raise ValueError("Event channel must be a string")
                for receiver in list(api.context.event_receivers):
                    try:
                        await receiver(channel, params.get("data"))
                    except (ConnectionError, OSError, RuntimeError) as exc:
                        api.log(f"Event delivery failed ({channel}): {exc}")
            elif method == "setSessionName":
                if api.context.session is None:
                    raise RuntimeError("Session persistence is disabled")
                api.context.session.rename(str(params["name"]))
            elif method == "setThinkingLevel":
                level = params["level"]
                if level not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
                    raise ValueError("Invalid thinking level")
                if api.context.agent.is_streaming:
                    raise RuntimeError("Changing thinking requires an idle agent")
                api.context.agent.thinking_level = level
                if api.context.session is not None:
                    api.context.session.append_thinking_change(level)
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
            await ui.set_widget(str(params.get("placement", "above")), list(params.get("lines") or []),
                                key=str(params.get("key") or ""))
            return None
        if call == "setHeader":
            await ui.set_header(list(params.get("lines") or []))
            return None
        if call == "setFooter":
            await ui.set_footer(list(params.get("lines") or []))
            return None
        if call == "setTitle":
            await ui.set_title(str(params.get("title", "")))
            return None
        raise ValueError(f"unsupported ui call: {call}")

    # ---- interactive custom views ------------------------------------------

    def _handle_custom_notification(self, message: dict) -> None:
        params = message.get("params") or {}
        session = self.custom_sessions.get(str(params.get("sessionId", "")))
        if session is None:
            return
        method = message.get("method")
        if method == "customDone":
            session.finish(params.get("result"))
        elif method == "customHandle":
            session.handle_action(str(params.get("action", "")))
        else:
            session.request_render()

    async def _open_custom(self, request: dict, params: dict) -> None:
        session_id = str(params.get("sessionId") or "")
        if not session_id:
            await self._reply(request, error="ui.custom requires a session id")
            return
        host = self.api.ui.custom_host
        if host is None:
            # Mirrors the TS RPC host: resolve immediately with no view.
            await self._reply(request, {"unavailable": True})
            return
        session = CustomViewSession(self, session_id, params.get("options") or {})
        self.custom_sessions[session_id] = session
        await self._reply(request, {"sessionId": session_id})
        task = asyncio.create_task(self._run_custom(host, session))
        self.custom_tasks.add(task)
        task.add_done_callback(self.custom_tasks.discard)

    def _handle_editor_notification(self, message: dict) -> None:
        session = self.editor_session
        if session is None:
            return
        params = message.get("params") or {}
        if params.get("sessionId") not in (None, session.id):
            return
        method = message.get("method")
        if method == "editorSubmit":
            session.submit(str(params.get("text", "")))
        elif method == "editorAction":
            session.request_action(str(params.get("action", "")))
        else:
            session.request_render()

    async def _open_editor(self, request: dict, params: dict) -> None:
        host = self.api.ui.editor_host
        if host is None:
            await self._reply(request, {"unavailable": True})
            return
        handler_id = params.get("handlerId")
        if not handler_id:
            if self.editor_session is not None:
                await self._notify("editorClose", {"sessionId": self.editor_session.id})
                self.editor_session = None
            outcome = host(None)
            if asyncio.iscoroutine(outcome):
                await outcome
            await self._reply(request, {})
            return
        if self.editor_session is not None:
            await self._notify("editorClose", {"sessionId": self.editor_session.id})
        self.editor_serial += 1
        session = CustomEditorSession(self, f"editor{self.editor_serial}")
        self.editor_session = session
        outcome = host(session)
        if asyncio.iscoroutine(outcome):
            await outcome
        await self._reply(request, {"sessionId": session.id})

    async def _run_custom(self, host, session: CustomViewSession) -> None:
        result = None
        try:
            result = await host(session)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - a broken view must not kill the bridge
            self.api.log(f"Custom view failed: {exc}")
        finally:
            self.custom_sessions.pop(session.id, None)
            await self._notify("customClose", {"sessionId": session.id, "result": to_jsonable(result)})

    # ---- parent-initiated dispatches ----------------------------------------

    def _context_payload(self) -> dict:
        context = self.api.context
        context.tool_catalog.update({tool.name: tool for tool in context.agent.tools})
        commands = {**context.commands, **self.api.commands}
        mcp_servers = context.mcp.registered_extension_servers() if context.mcp is not None else []
        return {"cwd": str(context.cwd), "hasSession": context.session is not None,
                "settings": context.config.settings if context.config else {},
                "commands": [{"name": name, "description": item[1], "source": "extension"} for name, item in commands.items()],
                "sessionName": context.session.name if context.session else None,
                "thinkingLevel": context.agent.thinking_level,
                "activeTools": [tool.name for tool in context.agent.tools],
                "allTools": [{"name": tool.name, "description": tool.description,
                              "parameters": tool.parameters} for tool in context.tool_catalog.values()],
                "mcpServers": mcp_servers}

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

    def _tool_renderer(self, handler_id: str):
        async def render(tool_name: str, phase: str, payload: dict, options: dict, width: int):
            result = await self._request("render", {"handlerId": handler_id, "kind": "tool",
                "toolName": tool_name, "phase": phase, "payload": to_jsonable(payload),
                "options": to_jsonable(options), "width": width,
                "context": {"cwd": str(self.api.context.cwd), **options}})
            return result.get("lines") if isinstance(result, dict) else None
        return render

    def _content_renderer(self, handler_id: str, kind: str):
        async def render(payload: dict, options: dict, width: int):
            result = await self._request("render", {"handlerId": handler_id, "kind": kind,
                "payload": to_jsonable(payload), "options": to_jsonable(options), "width": width})
            return result.get("lines") if isinstance(result, dict) else None
        return render

    def _markdown_transformer(self, handler_id: str):
        async def transform(markdown: str, context: dict):
            result = await self._request("render", {"handlerId": handler_id, "kind": "markdown",
                "markdown": markdown, "context": context})
            return result.get("markdown", markdown) if isinstance(result, dict) else markdown
        return transform

    def _virtual_model_route(self, handler_id: str):
        async def route(request: dict):
            return await self._request("routeVirtualModel", {"handlerId": handler_id,
                "request": to_jsonable(request), "context": self._context_payload()})
        return route

    async def set_flag_values(self, values: dict) -> None:
        await self._request("setFlags", {"values": values})

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
            if event_name in ("tool_call", "tool_result") and not isinstance(event, dict):
                wire_event = {"type": event_name, "toolCallId": event.tool_call_id,
                              "toolName": event.tool_name, "input": to_jsonable(event.args)}
                if event.parent_tool_call_id is not None:
                    wire_event["parentToolCallId"] = event.parent_tool_call_id
                if event_name == "tool_result":
                    wire_event.update(wire_message(event.result))
                    wire_event["isError"] = event.is_error
            else:
                wire_event = wire_message(event)
                wire_event["type"] = event_name
            payload = await self._request("dispatchEvent", {
                "handlerId": handler_id, "event": wire_event,
                "context": self._context_payload()})
            if not isinstance(payload, dict) or payload.get("error"):
                return None
            result = payload
            if event_name == "tool_result" and isinstance(result, dict):
                converted = result_from_json(result)
                return AfterToolCallResult(
                    content=converted.content if "content" in result else None,
                    details=converted.details, structured_content=converted.structured_content,
                    is_error=result.get("isError"), terminate=result.get("terminate"),
                    usage=usage_from_json(result["usage"]) if result.get("usage") is not None else None)
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

    async def _receive_bus_event(self, channel: str, data: object) -> None:
        if self.closed or self.process is None or self.process.stdin is None:
            return
        self.process.stdin.write((json.dumps({"method": "busEvent", "params": {
            "channel": channel, "data": data}}, ensure_ascii=False) + "\n").encode())
        await self.process.stdin.drain()

    def _unsubscribe_bus(self) -> None:
        receivers = self.api.context.event_receivers
        if self._receive_bus_event in receivers:
            receivers.remove(self._receive_bus_event)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._unsubscribe_bus()
        if self.editor_session is not None:
            await self._notify("editorClose", {"sessionId": self.editor_session.id})
            self.editor_session = None
        for task in list(self.custom_tasks):
            task.cancel()
        if self.custom_tasks:
            await asyncio.gather(*self.custom_tasks, return_exceptions=True)
        self.custom_sessions.clear()
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
