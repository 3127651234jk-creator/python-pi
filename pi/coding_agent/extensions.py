"""Python extension factories, command registry and ordered lifecycle hooks."""

import inspect
import importlib.util
import sys
import re
import uuid
from contextlib import redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..agent.agent import Agent
from ..agent.types import AgentTool, BeforeToolCallResult, AfterToolCallResult
from ..ai.providers import register_provider_stream_fn, unregister_provider_stream_fn
from .config import Configuration, read_json
from .resources import RESERVED_COMMANDS
from .session import SessionManager
from .ts_extension import InputResult, TsExtensionBridge, load_ts_extension

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

    async def set_widget(self, placement: str, lines: List[str]) -> None:
        if placement not in ("above", "below"):
            raise ValueError("widget placement must be above or below")
        await self._request("setWidget", placement=placement, lines=list(lines))

    async def set_title(self, title: str) -> None:
        await self._request("setTitle", title=title)


@dataclass
class ExtensionContext:
    agent: Agent
    cwd: Path
    session: Optional[SessionManager] = None


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
        self.keybindings: List[tuple] = []
        self.ui = ExtensionUI(path)
        self.sealed = False

    def check_registration(self):
        if self.sealed:
            raise RuntimeError("Register extension capabilities inside setup(api)")

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

    def send_message(self, custom_type: str, content: str, display: bool = True) -> Optional[str]:
        """Append a custom message that enters the LLM context."""
        if self.context.session is None:
            return None
        return self.context.session.append_custom_message(custom_type, content, display)

    def log(self, message: str):
        print(f"extension {self.path}: {message}", file=sys.stderr)


class ExtensionManager:
    def __init__(self, config: Configuration, agent: Agent, session: Optional[SessionManager] = None) -> None:
        self.config, self.agent = config, agent
        self.context = ExtensionContext(agent, config.cwd, session)
        self.extensions = []
        self.commands = {}
        self.module_names = []
        self.ts_bridges: List["TsExtensionBridge"] = []
        self.unsubscribe = None
        self.old_before, self.old_after = agent.before_tool_call, agent.after_tool_call
        self.output = None

    def paths(self):
        paths = []
        for base in (self.config.cwd / ".pi", self.config.agent_dir):
            for raw in read_json(base / "settings.json").get("extensions", []):
                path = Path(raw).expanduser()
                paths.append(path if path.is_absolute() else base / path)
        if self.config.settings.get("include_default_extensions", True):
            for base in (self.config.cwd / ".pi/extensions", self.config.agent_dir / "extensions"):
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
        await self.emit("session_start", {"type": "session_start"})

    def _seal(self, api: ExtensionAPI, path: Path) -> None:
        """Shared post-setup checks for Python and bridge-loaded extensions."""

        names = [tool.name for tool in api.tools]
        if len(set(names)) != len(names) or set(names) & {tool.name for tool in self.agent.tools}:
            raise ValueError("Extension tool name collision")
        if set(api.commands) & set(self.commands):
            raise ValueError("Extension command name collision")
        api.sealed = True
        self.extensions.append(api)
        self.agent.tools.extend(api.tools)
        self.commands.update(api.commands)

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
        try:
            bridge = await load_ts_extension(path, api)
            self.ts_bridges.append(bridge)
            self._seal(api, path)
        except Exception as exc:
            await self.cleanup_api(api)
            print(f"Extension {path} failed: {exc}", file=sys.stderr)

    async def cleanup_api(self, api):
        for handler in list(api.handlers.get("session_shutdown", [])):
            try:
                await invoke(handler, {"type": "session_shutdown"}, self.context)
            except Exception as exc:
                print(f"Extension cleanup failed: {exc}", file=sys.stderr)

    async def close(self):
        if self.unsubscribe:
            self.unsubscribe()
            self.unsubscribe = None
        for api in self.extensions:
            for name, config in api.providers.items():
                if callable(config.get("stream_simple")):
                    unregister_provider_stream_fn(name)
        for bridge in self.ts_bridges:
            try:
                await bridge.close()
            except Exception as exc:  # noqa: BLE001
                print(f"TS extension bridge shutdown failed: {exc}", file=sys.stderr)
        self.ts_bridges = []
        for api in reversed(self.extensions):
            await self.cleanup_api(api)
        owned = {id(tool) for api in self.extensions for tool in api.tools}
        self.agent.tools = [tool for tool in self.agent.tools if id(tool) not in owned]
        self.agent.before_tool_call, self.agent.after_tool_call = self.old_before, self.old_after
        for name in list(sys.modules):
            if any(name == root or name.startswith(root + ".") for root in self.module_names):
                sys.modules.pop(name, None)
        self.extensions, self.commands, self.module_names = [], {}, []
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
