"""Python extension factories, command registry and ordered lifecycle hooks."""

import asyncio
import inspect
import importlib.util
import sys
import re
import uuid
from contextlib import redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..agent.agent import Agent
from ..agent.types import AgentRequestUpdate, AgentTool, BeforeToolCallResult, AfterToolCallResult
from ..ai.providers import register_provider_stream_fn, unregister_provider_stream_fn
from .config import Configuration, read_json
from .model_registry import ModelRegistry
from .resources import RESERVED_COMMANDS
from .session import SessionManager
from .ts_extension import InputResult, TsExtensionBridge, load_ts_extension, to_jsonable

_extension_output = ContextVar("pi_extension_output", default=False)


class ExtensionOutput:
    def __init__(self, output):
        self.protocol_output = output

    def write(self, text):
        return (sys.stderr if _extension_output.get() else self.protocol_output).write(text)

    def flush(self):
        return (sys.stderr if _extension_output.get() else self.protocol_output).flush()

    def __getattr__(self, name):
        return getattr(self.protocol_output, name)


PROVIDER_FIELDS = {"api", "base_url", "api_key", "api_key_env", "headers", "models",
                   "oauth", "auth_required", "stream_simple"}


class ExtensionUI:
    """UI bridge for extensions: RPC request/response when a sink is wired,
    console fallback otherwise."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sink = None  # async callable(request: dict) -> response value
        # async callable(session) -> result; hosts ctx.ui.custom components.
        self.custom_host = None
        # sync callable() -> (columns, rows); sizes widget/header/footer views.
        self.size_provider = None
        # async/sync callable(session | None) -> None; installs ctx.ui.setEditorComponent.
        self.editor_host = None

    async def _request(self, method: str, **params):
        if self.sink is not None:
            return await self.sink({"method": method, **params})
        return None

    async def notify(self, message: str, level: str = "info") -> None:
        if self.sink is not None:
            await self._request("notify", message=message, level=level)
        else:
            print(f"extension {self.path}: {message}", file=sys.stderr)

    async def select(self, title: str, options: List[str]) -> Optional[str]:
        return await self._request("select", title=title, options=list(options))

    async def confirm(self, message: str) -> bool:
        return bool(await self._request("confirm", message=message))

    async def input(self, message: str, placeholder: str = "") -> str:
        value = await self._request("input", message=message, placeholder=placeholder)
        return value if isinstance(value, str) else ""

    async def set_status(self, text: str) -> None:
        await self._request("setStatus", text=text)

    async def set_widget(self, placement: str, lines: List[str], key: str = "") -> None:
        if placement in ("below", "belowEditor"):
            placement = "below"
        elif placement in ("above", "aboveEditor"):
            placement = "above"
        else:
            raise ValueError("widget placement must be above or below")
        await self._request("setWidget", placement=placement, lines=list(lines), key=key)

    async def set_header(self, lines: List[str]) -> None:
        await self._request("setHeader", lines=list(lines))

    async def set_footer(self, lines: List[str]) -> None:
        await self._request("setFooter", lines=list(lines))

    async def set_title(self, title: str) -> None:
        await self._request("setTitle", title=title)


@dataclass
class ExtensionContext:
    agent: Agent
    cwd: Path
    session: Optional[SessionManager] = None
    event_receivers: List[Callable] = field(default_factory=list, repr=False)
    config: Optional[Configuration] = None
    commands: dict = field(default_factory=dict)
    tool_catalog: dict = field(default_factory=dict)
    registry: Any = None
    mcp: Any = None
    refresh_tools: Optional[Callable] = None
    sync_providers: Optional[Callable] = None
    sync_virtual_models: Optional[Callable] = None
    mcp_servers_changed: Optional[Callable] = None


async def invoke(handler, *args):
    # Keep extension prints off the JSON/RPC protocol channel.
    token = _extension_output.set(True)
    try:
        result = handler(*args)
        return await result if inspect.isawaitable(result) else result
    except Exception as exc:
        raise RuntimeError(f"Extension callback failed: {exc}") from exc
    finally:
        _extension_output.reset(token)


class ExtensionAPI:
    def __init__(self, path: Path, context: ExtensionContext) -> None:
        self.path, self.context = path, context
        self.handlers: Dict[str, List[Callable]] = {}
        self.tools: List[AgentTool] = []
        self.commands = {}
        self.providers = {}
        self.virtual_models = {}
        self.mcp_servers = {}
        self.flags = {}
        self.flag_values = {}
        self.tool_renderers: List[Callable] = []
        self.message_renderers: Dict[str, Callable] = {}
        self.entry_renderers: Dict[str, Callable] = {}
        self.markdown_transformers: List[Callable] = []
        self.keybindings: List[tuple] = []
        self.ui = ExtensionUI(path)
        self.sealed = False
        self.closed = False

    def check_registration(self):
        if self.closed:
            raise RuntimeError("Extension is closed")

    def on(self, event: str, handler: Callable):
        self.check_registration()
        if not callable(handler):
            raise ValueError("Event handler must be callable")
        handlers = self.handlers.setdefault(event, [])
        handlers.append(handler)
        def unsubscribe():
            if handler in handlers:
                handlers.remove(handler)
        return unsubscribe

    def register_command(self, name: str, handler: Callable, description: str = ""):
        self.check_registration()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in RESERVED_COMMANDS or name == "extensions":
            raise ValueError("Invalid or reserved command name")
        if name in self.commands or not callable(handler):
            raise ValueError("Duplicate command or invalid handler")
        self.commands[name] = (handler, description)

    def register_tool(self, tool: AgentTool):
        self.check_registration()
        if not isinstance(tool, AgentTool) or not callable(tool.execute) or not re.fullmatch(r"[A-Za-z0-9_-]+", tool.name):
            raise ValueError("register_tool requires an executable AgentTool with a valid name")
        original = tool.execute
        async def execute(*args):
            return await invoke(original, *args)
        self.tools.append(replace(tool, execute=execute))

    async def execute_tool(self, name: str, args: Any = None, signal=None, on_update=None):
        """Run another tool as a child of the currently running tool call."""
        from .nested_tool_calls import execute_tool as nested_execute_tool

        return await nested_execute_tool(name, args, signal, on_update)

    def register_provider(self, name: str, config: dict):
        """Register a model provider in models.json provider format.

        A ``stream_simple(model, context, options)`` async generator implements a
        custom streaming protocol; it overrides dispatch for this provider's
        models. Other settings are validated and merged by the ModelRegistry on
        reload; user models.json entries override extension values for the same
        provider name.
        """
        self.check_registration()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not isinstance(config, dict):
            raise ValueError("register_provider requires a valid name and a config object")
        unknown = set(config) - PROVIDER_FIELDS
        if unknown:
            raise ValueError(f"Unknown provider settings: {', '.join(sorted(unknown))}")
        if "stream_simple" in config and not callable(config["stream_simple"]):
            raise ValueError("stream_simple must be an async generator function")
        self.providers[name] = config
        if callable(config.get("stream_simple")):
            register_provider_stream_fn(name, config["stream_simple"])
        if self.sealed and self.context.sync_providers:
            self.context.sync_providers()

    def unregister_provider(self, name: str):
        self.check_registration()
        self.providers.pop(name, None)
        unregister_provider_stream_fn(name)
        if self.sealed and self.context.sync_providers:
            self.context.sync_providers()

    def register_flag(self, name: str, options: dict):
        self.check_registration()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in self.flags:
            raise ValueError("Invalid or duplicate extension flag")
        if not isinstance(options, dict) or options.get("type") not in ("boolean", "string"):
            raise ValueError("Extension flag type must be boolean or string")
        default = options.get("default")
        expected = bool if options["type"] == "boolean" else str
        if default is not None and type(default) is not expected:
            raise ValueError("Extension flag default has the wrong type")
        self.flags[name] = dict(options)
        if default is not None:
            self.flag_values[name] = default

    def get_flag(self, name: str):
        return self.flag_values.get(name)

    def register_tool_renderer(self, renderer: Callable):
        self.check_registration()
        self.tool_renderers.append(renderer)

    def register_message_renderer(self, custom_type: str, renderer: Callable):
        self.check_registration()
        self.message_renderers[custom_type] = renderer

    def register_entry_renderer(self, custom_type: str, renderer: Callable):
        self.check_registration()
        self.entry_renderers[custom_type] = renderer

    def register_markdown_transformer(self, transformer: Callable):
        self.check_registration()
        self.markdown_transformers.append(transformer)

    def register_virtual_model(self, definition: dict):
        self.check_registration()
        provider, model_id = definition.get("provider"), definition.get("id")
        if not isinstance(provider, str) or not provider or not isinstance(model_id, str) or not model_id:
            raise ValueError("Virtual model requires provider and id")
        if not callable(definition.get("route")):
            raise ValueError("Virtual model requires a route function")
        self.virtual_models[(provider, model_id)] = definition
        if self.sealed and self.context.sync_virtual_models:
            self.context.sync_virtual_models()

    def unregister_virtual_model(self, provider: str, model_id: str):
        self.check_registration()
        self.virtual_models.pop((provider, model_id), None)
        if self.sealed and self.context.sync_virtual_models:
            self.context.sync_virtual_models()

    async def register_mcp_server(self, name: str, config: dict):
        self.check_registration()
        if self.context.mcp is None:
            raise RuntimeError("MCP manager is unavailable")
        await self.context.mcp.register_extension_server(name, config, str(self.path))
        self.mcp_servers[name] = dict(config)
        if self.context.mcp_servers_changed:
            await self.context.mcp_servers_changed()

    async def unregister_mcp_server(self, name: str):
        self.check_registration()
        if self.context.mcp is None:
            raise RuntimeError("MCP manager is unavailable")
        await self.context.mcp.unregister_extension_server(name, str(self.path))
        self.mcp_servers.pop(name, None)
        if self.context.mcp_servers_changed:
            await self.context.mcp_servers_changed()

    def get_mcp_servers(self) -> List[dict]:
        if self.context.mcp is None:
            return []
        return self.context.mcp.registered_extension_servers()

    def refresh_tools(self) -> None:
        if self.context.refresh_tools:
            self.context.refresh_tools()

    def register_keybinding(self, action: str, keys: List[str], handler: Callable):
        """Register a terminal shortcut; the handler runs when the editor is idle."""
        self.check_registration()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", action) or not callable(handler):
            raise ValueError("register_keybinding requires an action name and callable handler")
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError("keybinding requires a nonempty list of shortcuts")
        self.keybindings.append((action, keys, handler))

    def append_entry(self, custom_type: str, data: dict) -> Optional[str]:
        """Persist extension state as a session custom entry."""
        if self.context.session is None:
            return None
        return self.context.session.append_custom(custom_type, data)

    def send_message(self, custom_type: str, content, display: bool = True,
                     details=None) -> Optional[str]:
        """Append a custom message that enters the LLM context."""
        if self.context.session is None:
            return None
        return self.context.session.append_custom_message(custom_type, content, display, details)

    def log(self, message: str):
        print(f"extension {self.path}: {message}", file=sys.stderr)


class ExtensionManager:
    def __init__(self, config: Configuration, agent: Agent, session: Optional[SessionManager] = None,
                 registry=None, mcp=None, flag_values: Optional[dict] = None) -> None:
        self.config, self.agent = config, agent
        self.context = ExtensionContext(agent, config.cwd, session)
        self.context.config = config
        self.context.registry = registry
        self.context.mcp = mcp
        self.context.refresh_tools = self.refresh_tools
        self.context.sync_providers = self.sync_providers
        self.context.sync_virtual_models = self.sync_virtual_models
        self.context.mcp_servers_changed = self.mcp_servers_changed
        self.flag_values = dict(flag_values or {})
        self.extensions = []
        self.commands = {}
        self.module_names = []
        self.ts_bridges: List["TsExtensionBridge"] = []
        self.unsubscribe = None
        self.old_before, self.old_after = agent.before_tool_call, agent.after_tool_call
        self.old_prepare_request = agent.prepare_request
        self.output = None
        self.custom_host = None
        self.size_provider = None
        self.editor_host = None

    def refresh_tools(self) -> None:
        """Publish tools registered after setup without changing the active set otherwise."""
        known = {tool.name: tool for tool in self.agent.tools}
        for api in self.extensions:
            for tool in api.tools:
                known[tool.name] = tool
        self.agent.tools[:] = list(known.values())
        self.context.tool_catalog.update(known)

    def sync_providers(self) -> None:
        if self.context.registry is not None:
            self.context.registry.register_extension_providers(self.all_providers())

    def sync_virtual_models(self) -> None:
        if self.context.registry is not None:
            self.context.registry.register_virtual_models(self.all_virtual_models())

    async def mcp_servers_changed(self) -> None:
        servers = self.context.mcp.registered_extension_servers() if self.context.mcp is not None else []
        asyncio.create_task(self.emit("mcp_servers_change", {
            "type": "mcp_servers_change", "servers": servers}))

    async def apply_flag_values(self) -> None:
        registered = {name: options for api in self.extensions for name, options in api.flags.items()}
        unknown = sorted(set(self.flag_values) - set(registered))
        if unknown:
            raise ValueError("Unknown option" + ("s" if len(unknown) > 1 else "") + ": "
                             + ", ".join("--" + name for name in unknown))
        values = {}
        for name, options in registered.items():
            value = self.flag_values.get(name, options.get("default"))
            if options["type"] == "boolean":
                value = name in self.flag_values or bool(value)
            elif name in self.flag_values and not isinstance(value, str):
                raise ValueError(f'Extension flag "--{name}" requires a value')
            values[name] = value
        for api in self.extensions:
            api.flag_values.update({name: value for name, value in values.items() if name in api.flags})
        for bridge in self.ts_bridges:
            await bridge.set_flag_values(values)

    def all_virtual_models(self) -> dict:
        models = {}
        for api in self.extensions:
            models.update(api.virtual_models)
        return models

    async def transform_markdown(self, markdown: str, message_type: str,
                                 is_streaming: bool = False, available_width: int = 80) -> str:
        value = markdown
        context = {"messageType": message_type, "isStreaming": is_streaming,
                   "availableWidth": available_width}
        for api in self.extensions:
            for transformer in api.markdown_transformers:
                value = await invoke(transformer, value, context)
        return value

    async def render_tool(self, tool_name: str, phase: str, payload: dict,
                          options: Optional[dict] = None, width: int = 80) -> Optional[List[str]]:
        for api in self.extensions:
            for renderer in api.tool_renderers:
                lines = await invoke(renderer, tool_name, phase, payload, options or {}, width)
                if lines is not None:
                    return lines
        return None

    async def render_message(self, custom_type: str, message: dict,
                             options: Optional[dict] = None, width: int = 80) -> Optional[List[str]]:
        for api in reversed(self.extensions):
            renderer = api.message_renderers.get(custom_type)
            if renderer:
                return await invoke(renderer, message, options or {}, width)
        return None

    async def render_entry(self, custom_type: str, entry: dict,
                           options: Optional[dict] = None, width: int = 80) -> Optional[List[str]]:
        for api in reversed(self.extensions):
            renderer = api.entry_renderers.get(custom_type)
            if renderer:
                return await invoke(renderer, entry, options or {}, width)
        return None

    def paths(self):
        paths = []
        bases = (self.config.cwd / ".pi", self.config.agent_dir) if self.config.project_trusted \
            else (self.config.agent_dir,)
        for base in bases:
            for raw in read_json(base / "settings.json").get("extensions", []):
                path = Path(raw).expanduser()
                paths.append(path if path.is_absolute() else base / path)
        if self.config.settings.get("include_default_extensions", True):
            default_bases = [self.config.agent_dir / "extensions"]
            if self.config.project_trusted:
                default_bases.insert(0, self.config.cwd / ".pi/extensions")
            for base in default_bases:
                if base.is_dir():
                    paths.extend(sorted(base.iterdir()))
        result = []
        for path in paths:
            if path.is_dir():
                path = path / "__init__.py"
            if path.suffix in (".py", ".ts", ".mts", ".js", ".mjs") and path.exists() \
                    and path.resolve() not in result:
                result.append(path.resolve())
        return result

    async def emit(self, name, event, strict=False):
        results = []
        handlers = [handler for api in self.extensions for handler in list(api.handlers.get(name, []))]
        for handler in handlers:
            try:
                results.append(await invoke(handler, event, self.context))
            except Exception as exc:
                if strict:
                    raise
                print(f"Extension event {name} failed: {exc}", file=sys.stderr)
        return results

    async def process_input(self, text: str) -> Optional[str]:
        handlers = [handler for api in self.extensions for handler in list(api.handlers.get("input", []))]
        for handler in handlers:
            result = await invoke(handler, {"text": text}, self.context)
            if isinstance(result, InputResult):
                if result.handled:
                    return None
                if result.text is not None:
                    text = result.text
        command, _, arguments = text.partition(" ")
        if command.startswith("/") and command[1:] in self.commands:
            handler, _ = self.commands[command[1:]]
            result = await invoke(handler, arguments, self.context)
            if result is not None and not isinstance(result, str):
                raise ValueError("Command handler must return a prompt string or None")
            return result
        return text

    async def reload(self):
        if self.agent.is_streaming:
            raise RuntimeError("Reload requires an idle agent")
        await self.close()
        self.output = ExtensionOutput(sys.stdout)
        sys.stdout = self.output
        for path in self.paths():
            if path.suffix in (".ts", ".mts", ".js", ".mjs"):
                await self._load_remote(path)
            else:
                await self._load_python(path)
        await self.apply_flag_values()
        self.sync_providers()
        self.sync_virtual_models()
        async def event(event):
            await self.emit(event.type, event)
        self.unsubscribe = self.agent.subscribe(event)

        async def before(context, signal):
            if self.old_before:
                result = await self.old_before(context, signal)
                if result and result.block:
                    return result
            for result in await self.emit("tool_call", context, strict=True):
                if isinstance(result, BeforeToolCallResult) and result.block:
                    return result
            return None

        async def after(context, signal):
            result = await self.old_after(context, signal) if self.old_after else None
            for current in await self.emit("tool_result", context, strict=True):
                if isinstance(current, AfterToolCallResult):
                    result = current
            return result

        self.agent.before_tool_call = before
        self.agent.after_tool_call = after

        async def prepare_request(request, signal):
            if self.old_prepare_request:
                previous = self.old_prepare_request(request, signal)
                previous = await previous if inspect.isawaitable(previous) else previous
                if previous is not None:
                    request = replace(request,
                        context=previous.context or request.context,
                        model=previous.model or request.model,
                        thinking_level=previous.thinking_level or request.thinking_level)
            selected = self.agent.model
            definition = self.all_virtual_models().get((selected.provider, selected.id))
            if definition is None:
                return None
            state = None
            if self.context.session is not None:
                saved = self.context.session.latest_custom("pi.virtual-model-state")
                if saved and saved.get("provider") == selected.provider and saved.get("modelId") == selected.id:
                    state = saved.get("state")
            messages = request.context.messages
            reason = "user" if messages and getattr(messages[-1], "role", None) == "user" else "continuation"
            routed = await definition["route"]({
                "model": {"provider": selected.provider, "id": selected.id, "name": selected.name,
                          "api": selected.api},
                "thinkingLevel": request.thinking_level, "reason": reason,
                "state": state, "messages": to_jsonable(messages),
            })
            if not isinstance(routed, dict) or not isinstance(routed.get("model"), dict):
                raise ValueError("Virtual model route must return a model")
            target = routed["model"]
            registry = self.context.registry
            if registry is None:
                registry = ModelRegistry(self.config)
            physical = registry.resolve(target["provider"], target["id"])
            if physical.api == "pi-virtual":
                raise ValueError("Virtual model routes must select a physical model")
            if "state" in routed and self.context.session is not None and routed["state"] is not state:
                self.context.session.append_custom("pi.virtual-model-state", {
                    "provider": selected.provider, "modelId": selected.id, "state": routed["state"]})
            return AgentRequestUpdate(model=physical,
                thinking_level=routed.get("thinkingLevel", request.thinking_level))

        self.agent.prepare_request = prepare_request
        await self.emit("session_start", {"type": "session_start"})

    def _seal(self, api: ExtensionAPI, path: Path) -> None:
        """Shared post-setup checks for Python and bridge-loaded extensions."""

        names = [tool.name for tool in api.tools]
        if len(set(names)) != len(names) or set(names) & {tool.name for tool in self.agent.tools}:
            raise ValueError("Extension tool name collision")
        if set(api.commands) & set(self.commands):
            raise ValueError("Extension command name collision")
        api.sealed = True
        api.ui.custom_host = self.custom_host
        api.ui.size_provider = self.size_provider
        api.ui.editor_host = self.editor_host
        self.extensions.append(api)
        self.agent.tools.extend(api.tools)
        self.commands.update(api.commands)
        self.context.commands = self.commands

    async def _load_python(self, path: Path) -> None:
        name = "_pi_extension_" + uuid.uuid4().hex
        api = ExtensionAPI(path, self.context)
        try:
            spec = importlib.util.spec_from_file_location(name, path,
                submodule_search_locations=[str(path.parent)] if path.name == "__init__.py" else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Compile fresh source so rapid reloads cannot reuse stale .pyc files.
            with redirect_stdout(sys.stderr):
                exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
            setup = getattr(module, "setup", None)
            if not callable(setup):
                raise ValueError("Extension must define setup(api)")
            await invoke(setup, api)
            self._seal(api, path)
            self.module_names.append(name)
        except Exception as exc:
            await self.cleanup_api(api)
            for module_name in list(sys.modules):
                if module_name == name or module_name.startswith(name + "."):
                    sys.modules.pop(module_name, None)
            print(f"Extension {path} failed: {exc}", file=sys.stderr)

    async def _load_remote(self, path: Path) -> None:
        """Load a TS/JS extension through the Node bridge into a shared ExtensionAPI."""

        api = ExtensionAPI(path, self.context)
        bridge = None
        try:
            bridge = await load_ts_extension(path, api)
            self._seal(api, path)
            self.ts_bridges.append(bridge)
        except Exception as exc:
            await self.cleanup_api(api)
            if bridge is not None:
                await bridge.close()
            print(f"Extension {path} failed: {exc}", file=sys.stderr)

    async def cleanup_api(self, api):
        for handler in list(api.handlers.get("session_shutdown", [])):
            try:
                await invoke(handler, {"type": "session_shutdown"}, self.context)
            except Exception as exc:
                print(f"Extension cleanup failed: {exc}", file=sys.stderr)
        api.closed = True

    async def close(self):
        if self.unsubscribe:
            self.unsubscribe()
            self.unsubscribe = None
        for api in reversed(self.extensions):
            await self.cleanup_api(api)
        for api in self.extensions:
            for name, config in api.providers.items():
                if callable(config.get("stream_simple")):
                    unregister_provider_stream_fn(name)
            if self.context.mcp is not None:
                for name in list(api.mcp_servers):
                    await self.context.mcp.unregister_extension_server(name, str(api.path))
        for bridge in self.ts_bridges:
            try:
                await bridge.close()
            except Exception as exc:  # noqa: BLE001
                print(f"TS extension bridge shutdown failed: {exc}", file=sys.stderr)
        self.ts_bridges = []
        owned = {id(tool) for api in self.extensions for tool in api.tools}
        active_names = {tool.name for tool in self.agent.tools}
        self.agent.tools.extend(tool for name, tool in self.context.tool_catalog.items()
                                if name not in active_names and id(tool) not in owned)
        self.agent.tools = [tool for tool in self.agent.tools if id(tool) not in owned]
        self.agent.before_tool_call, self.agent.after_tool_call = self.old_before, self.old_after
        self.agent.prepare_request = self.old_prepare_request
        for name in list(sys.modules):
            if any(name == root or name.startswith(root + ".") for root in self.module_names):
                sys.modules.pop(name, None)
        self.extensions, self.commands, self.module_names = [], {}, []
        self.context.commands = self.commands
        self.context.tool_catalog.clear()
        self.sync_providers()
        self.sync_virtual_models()
        if self.output is not None and sys.stdout is self.output:
            sys.stdout = self.output.protocol_output
        self.output = None

    def all_providers(self) -> dict:
        """Merged extension provider configs; earlier extensions lose to later ones."""
        providers = {}
        for api in self.extensions:
            providers.update(api.providers)
        return providers

    def all_keybindings(self) -> List[tuple]:
        return [(action, keys, handler) for api in self.extensions for action, keys, handler in api.keybindings]

    def set_ui_sink(self, sink) -> None:
        """Route extension UI requests through a client (RPC mode)."""
        for api in self.extensions:
            api.ui.sink = sink

    def set_editor_host(self, host) -> None:
        """Install extension ``ctx.ui.setEditorComponent`` editors on the TUI."""
        self.editor_host = host
        for api in self.extensions:
            api.ui.editor_host = host

    def set_custom_host(self, host, size=None) -> None:
        """Route ``ctx.ui.custom`` components to an interactive host.

        Existing and later-loaded extensions both see the host. ``size`` is a
        sync callable returning the terminal ``(columns, rows)`` used to render
        widget/header/footer component factories. Without a host the bridge
        resolves ``ui.custom`` immediately, like the TS RPC host.
        """
        self.custom_host = host
        self.size_provider = size
        for api in self.extensions:
            api.ui.custom_host = host
            api.ui.size_provider = size
