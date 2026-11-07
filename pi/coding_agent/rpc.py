"""LF-delimited JSON RPC server matching TypeScript ``pi --mode rpc``.

Wire protocol (authoritative: ``packages/coding-agent/src/modes/rpc``):

- stdin commands are ``{id?, type, ...}`` objects with TS field names
  (``modelId``, ``customInstructions``, ``sessionPath``, ``entryId``, ...).
- stdout responses are ``{id?, type: "response", command, success, data?|error}``;
  ``data`` is omitted when the command has none. Parse failures answer with
  ``command: "parse"`` and unknown commands fail with ``Unknown command: ...``.
- Agent activity streams as TS-shaped events (``rpc_wire.event_to_ts``);
  ``message_update`` carries ``{usage, assistantMessageEvent}`` with the
  cumulative partial snapshot stripped, and every run ends with
  ``agent_settled``. Queue changes also emit ``queue_update``.
- Extension UI requests are remapped to the TS method/field names
  (``extension_ui_request`` on stdout, ``extension_ui_response`` on stdin).

Known limitations versus TS: ``bash``/``abort_bash``, ``set_auto_retry``/
``abort_retry`` and ``new_session.parentSession`` are rejected with explicit
errors (no Python runtime support); TS session-level events
(``entry_appended``, ``compaction_start/end``, ``auto_retry_*``) are not
emitted; the Python agent loop consumes the stream's terminal ``done``/``error``
events, so those never appear as ``message_update`` inner events (the final
message arrives through ``message_end``); ``steeringMode``/``followUpMode``
both reflect the single Python ``queue_mode``; extension commands carry
``sourceInfo`` only when their originating extension file is known.

Python-only commands kept for this port: ``get_tools``, ``get_usage``,
``sync_models``, ``reload_extensions``.
"""

import asyncio
import json
import os
import sys
import uuid
from typing import Optional

from ..ai.json_io import to_json
from ..ai.models import ModelError
from ..ai.types import AssistantMessage, ImageContent, TextContent, get_message_text
from ..agent.agent import Agent
from ..agent.compaction import compact_agent, restore_compaction
from .catalog_sync import fetch_catalog, save_cache
from .extensions import ExtensionManager
from .model_registry import ModelRegistry
from .resources import ResourceCatalog
from .rpc_wire import (
    compaction_to_ts,
    entry_to_ts,
    event_to_ts,
    model_to_ts,
    stats_to_ts,
    tree_nodes_to_ts,
)
from .session import SessionManager
from .ts_interop import our_message_to_ts
from .usage import session_stats, summary_usage_extra, usage_cost_breakdown

_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


class JsonWriter:
    def __init__(self, output=None) -> None:
        self.output = output or getattr(sys.stdout, "protocol_output", sys.stdout)
        self.lock = asyncio.Lock()

    async def write(self, record):
        line = json.dumps(to_json(record), ensure_ascii=False, allow_nan=False) + "\n"
        async with self.lock:
            await asyncio.to_thread(self.output.write, line)
            await asyncio.to_thread(self.output.flush)

    async def __call__(self, event):
        await self.write(event)


def _images_from_command(raw) -> Optional[list]:
    """TS ImageContent input ({type: "image", data, mimeType}) -> dataclasses."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("images must be an array")
    images = []
    for item in raw:
        if (not isinstance(item, dict) or not isinstance(item.get("data"), str)
                or not isinstance(item.get("mimeType"), str)):
            raise ValueError("each image requires string data and mimeType")
        images.append(ImageContent(data=item["data"], mime_type=item["mimeType"]))
    return images or None


def _session_id(session_file: Optional[str]) -> str:
    if not session_file:
        return ""
    base = os.path.basename(session_file)
    return base[:-len(".jsonl")] if base.endswith(".jsonl") else base


def _ui_request_to_ts(request: dict) -> dict:
    """Remap the Python extension UI payload to the TS request shape."""
    method = request.get("method")
    mapped = dict(request)
    if method == "notify":
        mapped["notifyType"] = mapped.pop("level", "info")
    elif method == "setStatus":
        mapped["statusKey"] = mapped.pop("key", "")
        mapped["statusText"] = mapped.pop("text", None)
    elif method == "setWidget":
        mapped["widgetKey"] = mapped.pop("key", "")
        mapped["widgetLines"] = mapped.pop("lines", None)
        placement = mapped.pop("placement", None)
        if placement is not None:
            mapped["widgetPlacement"] = {"above": "aboveEditor", "below": "belowEditor"}.get(placement, placement)
    elif method == "input":
        mapped["title"] = mapped.pop("message", "")
        placeholder = mapped.pop("placeholder", "")
        if placeholder:
            mapped["placeholder"] = placeholder
    elif method == "confirm":
        mapped.setdefault("title", "")
    return mapped


class RpcServer:
    def __init__(self, agent: Agent, writer: JsonWriter, session: Optional[SessionManager] = None,
                 registry: Optional[ModelRegistry] = None, resources: Optional[ResourceCatalog] = None,
                 extensions: Optional[ExtensionManager] = None) -> None:
        self.agent, self.writer, self.session = agent, writer, session
        self.registry, self.resources = registry, resources
        self.task = None
        self.extensions = extensions
        self.ui_futures: dict = {}

    # -- extension UI protocol ------------------------------------------------

    async def ui_sink(self, request: dict):
        """Serve one extension UI request; resolved by extension_ui_response."""
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.ui_futures[request_id] = future
        try:
            payload = {"type": "extension_ui_request", "id": request_id}
            payload.update(_ui_request_to_ts(request))
            await self.writer.write(payload)
            reply = await future
            return reply.get("value") if isinstance(reply, dict) else None
        finally:
            self.ui_futures.pop(request_id, None)

    def resolve_ui(self, reply: dict) -> None:
        future = self.ui_futures.get(reply.get("id"))
        if future is not None and not future.done():
            future.set_result(reply)

    # -- event stream ---------------------------------------------------------

    async def run_prompt(self, text, images=None):
        async def event(event):
            if self.session and event.type == "message_end":
                self.session.append(event.message)
            await self.writer.write(event_to_ts(event))
        unsubscribe = self.agent.subscribe(event)
        try:
            await self.agent.prompt(text, images=images)
        except Exception as exc:
            await self.writer.write({"type": "run_error", "error": str(exc)})
        finally:
            unsubscribe()
            await self.writer.write({"type": "agent_settled"})

    def busy(self):
        return self.task is not None and not self.task.done()

    def queue_texts(self, queue):
        return [get_message_text(message.content) for message in queue]

    async def emit_queue_update(self) -> None:
        await self.writer.write({"type": "queue_update",
                                 "steering": self.queue_texts(self.agent._steering_queue),
                                 "followUp": self.queue_texts(self.agent._follow_up_queue)})

    def _require_idle(self) -> None:
        if self.busy():
            raise ValueError("Command requires an idle agent")

    def _require_session(self) -> SessionManager:
        if self.session is None:
            raise ValueError("Session support is disabled")
        return self.session

    def _require_registry(self) -> ModelRegistry:
        if self.registry is None:
            raise ValueError("No model registry available")
        return self.registry

    async def dispatch(self, command):
        kind = command.get("type") if isinstance(command, dict) else None
        response = {"type": "response", "command": kind if isinstance(kind, str) else "parse", "success": True}
        if isinstance(command, dict) and "id" in command:
            response["id"] = command["id"]
        try:
            if "id" in command and not isinstance(command["id"], str):
                raise ValueError("id must be a string")
            if not isinstance(kind, str):
                raise ValueError("type must be a string")
            if await self._handle(kind, command, response):
                return
        except (KeyError, ValueError, TypeError, OSError, RuntimeError, ModelError) as exc:
            response.pop("data", None)
            response.update(success=False, error=str(exc))
        await self.writer.write(response)

    async def _handle(self, kind: str, command: dict, response: dict) -> bool:
        """Apply one command; returns True when the response was already written."""
        if kind in ("prompt", "steer", "follow_up"):
            text = command.get("message")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("message must be a nonempty string")
            images = _images_from_command(command.get("images"))
            if self.extensions and kind == "prompt":
                if self.busy():
                    raise ValueError("Agent is busy")
                text = await self.extensions.process_input(text)
                if text is None:
                    response["data"] = {"disposition": "handled"}
                    await self.writer.write(response)
                    return True
            if self.resources:
                text = self.resources.expand(text)
            if kind == "prompt":
                if self.busy():
                    behavior = command.get("streamingBehavior")
                    if behavior not in ("steer", "followUp"):
                        raise ValueError("Agent is busy; specify streamingBehavior "
                                         "('steer' or 'followUp') to queue the message")
                    if behavior == "steer":
                        self.agent.steer(text, images)
                    else:
                        self.agent.follow_up(text, images)
                    await self.emit_queue_update()
                    response["data"] = {"disposition": "queued"}
                    await self.writer.write(response)
                    return True
                response["data"] = {"disposition": "started"}
                await self.writer.write(response)
                self.task = asyncio.create_task(self.run_prompt(text, images))
                return True
            if kind == "steer":
                self.agent.steer(text, images)
            else:
                self.agent.follow_up(text, images)
            response["data"] = {"disposition": "queued"}
            await self.emit_queue_update()
            return False

        if kind == "abort":
            self.agent.abort()
        elif kind == "clear_queue":
            steering = self.queue_texts(self.agent._steering_queue)
            follow_up = self.queue_texts(self.agent._follow_up_queue)
            self.agent._steering_queue.clear()
            self.agent._follow_up_queue.clear()
            response["data"] = {"steering": steering, "followUp": follow_up}
            await self.emit_queue_update()
        elif kind == "set_steering_mode" or kind == "set_follow_up_mode":
            mode = command["mode"]
            if mode not in ("all", "one-at-a-time"):
                raise ValueError("mode must be all or one-at-a-time")
            self.agent.queue_mode = mode
        elif kind == "get_state":
            data = {}
            if self.agent.model is not None:
                data["model"] = model_to_ts(self.agent.model)
            data["thinkingLevel"] = self.agent.thinking_level
            data["isStreaming"] = self.busy()
            data["isCompacting"] = False
            data["steeringMode"] = self.agent.queue_mode
            data["followUpMode"] = self.agent.queue_mode
            if self.session is not None and self.session.session_file:
                data["sessionFile"] = self.session.session_file
                data["sessionId"] = _session_id(self.session.session_file)
            else:
                data["sessionId"] = ""
            if self.session is not None and self.session.name:
                data["sessionName"] = self.session.name
            data["autoCompactionEnabled"] = self.agent.compactor.settings.enabled
            data["messageCount"] = len(self.agent.messages)
            data["pendingMessageCount"] = len(self.agent._steering_queue) + len(self.agent._follow_up_queue)
            response["data"] = data
        elif kind == "get_session_stats":
            session_file = self.session.session_file if self.session else None
            extra = summary_usage_extra(self.session) if self.session else None
            response["data"] = stats_to_ts(
                session_stats(self.agent.messages, self.agent.model, session_file, extra_usage=extra),
                _session_id(session_file))
        elif kind == "get_usage":
            totals = usage_cost_breakdown(self.agent.messages, self.registry,
                                          extra_usage=summary_usage_extra(self.session) if self.session else None)
            response["data"] = {
                "requests": totals.requests, "input": totals.input, "output": totals.output,
                "cacheRead": totals.cache_read, "cacheWrite": totals.cache_write,
                "cost": totals.cost,
                "byModel": {key: {"requests": entry.requests, "input": entry.input,
                                  "output": entry.output, "cacheRead": entry.cache_read,
                                  "cacheWrite": entry.cache_write, "cost": entry.cost}
                            for key, entry in totals.by_model.items()},
            }
        elif kind == "get_last_assistant_text":
            texts = [block.text for message in self.agent.messages if isinstance(message, AssistantMessage)
                     for block in message.content if isinstance(block, TextContent)]
            response["data"] = {"text": texts[-1] if texts else None}
        elif kind == "get_available_thinking_levels":
            response["data"] = {"levels": list(_THINKING_LEVELS)}
        elif kind == "set_thinking_level":
            self._require_idle()
            level = command["level"]
            if level not in _THINKING_LEVELS:
                raise ValueError("Invalid thinking level")
            self.agent.thinking_level = level
        elif kind == "cycle_thinking_level":
            self._require_idle()
            index = _THINKING_LEVELS.index(self.agent.thinking_level) \
                if self.agent.thinking_level in _THINKING_LEVELS else -1
            level = _THINKING_LEVELS[(index + 1) % len(_THINKING_LEVELS)]
            self.agent.thinking_level = level
            response["data"] = {"level": level}
        elif kind == "get_available_models":
            registry = self._require_registry()
            response["data"] = {"models": [model_to_ts(model) for model in registry.models.values()]}
        elif kind == "set_model":
            registry = self._require_registry()
            self._require_idle()
            model_id = command.get("modelId")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("set_model requires provider and modelId")
            self.agent.model = registry.resolve(command.get("provider", self.agent.model.provider), model_id)
            response["data"] = model_to_ts(self.agent.model)
        elif kind == "cycle_model":
            registry = self._require_registry()
            self._require_idle()
            models = list(registry.models.values())
            if not models:
                response["data"] = None
            else:
                current = (self.agent.model.provider, self.agent.model.id)
                index = next((i for i, model in enumerate(models)
                              if (model.provider, model.id) == current), -1)
                self.agent.model = models[(index + 1) % len(models)]
                response["data"] = {"model": model_to_ts(self.agent.model),
                                    "thinkingLevel": self.agent.thinking_level, "isScoped": False}
        elif kind == "compact":
            self._require_registry()
            if self.busy():
                raise ValueError("Cannot compact while the agent is streaming")
            result = await compact_agent(self.agent, self.session, command.get("customInstructions", ""))
            response["data"] = compaction_to_ts(result)
        elif kind == "set_auto_compaction":
            self._require_idle()
            enabled = command["enabled"]
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be a boolean")
            self.agent.compactor.settings.enabled = enabled
        elif kind in ("set_auto_retry", "abort_retry", "bash", "abort_bash"):
            raise ValueError(f"{kind} is not supported by the Python RPC server")
        elif kind == "new_session":
            self._require_idle()
            if command.get("parentSession") is not None:
                raise ValueError("parentSession is not supported by the Python RPC server")
            self.agent.messages = [m for m in self.agent.messages if isinstance(m, SystemMessage)]
            if self.session:
                self.session.start_new()
                for message in self.agent.messages:
                    self.session.append(message)
            restore_compaction(self.agent.compactor, self.agent.messages,
                               self.session.latest_custom("compaction") if self.session else None)
            response["data"] = {"cancelled": False}
        elif kind == "get_messages":
            response["data"] = {"messages": [our_message_to_ts(m) for m in self.agent.messages]}
        elif kind == "get_tools":
            response["data"] = {"tools": [{"name": t.name, "description": t.description,
                                           "parameters": t.parameters} for t in self.agent.tools]}
        elif kind == "get_commands":
            commands = []
            if self.extensions:
                owner = {}
                for api in self.extensions.extensions:
                    for name in api.commands:
                        owner[name] = api.path
                agent_dir = str(self.extensions.config.agent_dir)
                for name, (_, description) in self.extensions.commands.items():
                    record = {"name": name, "description": description, "source": "extension"}
                    path = owner.get(name)
                    if path is not None:
                        path_text = str(path)
                        record["sourceInfo"] = {
                            "path": path_text,
                            "source": "extension",
                            "scope": "user" if path_text.startswith(agent_dir) else "project",
                            "origin": "top-level",
                        }
                    commands.append(record)
            response["data"] = {"commands": commands}
        elif kind == "sync_models":
            self._require_idle()
            registry = self._require_registry()
            providers = await fetch_catalog()
            save_cache(registry.config.agent_dir, providers)
            registry.reload()
            response["data"] = {"providers": sorted(providers)}
        elif kind == "reload_extensions":
            self._require_idle()
            if self.extensions:
                await self.extensions.reload()
        elif kind in ("switch_session", "fork", "clone", "set_session_name", "export_html"):
            session = self._require_session()
            self._require_idle()
            if kind == "switch_session":
                path = command.get("sessionPath")
                if not isinstance(path, str) or not path:
                    raise ValueError("switch_session requires sessionPath")
                self.agent.messages = session.resume(path)
                restore_compaction(self.agent.compactor, self.agent.messages,
                                   session.latest_custom("compaction"))
                response["data"] = {"cancelled": False}
            elif kind == "fork":
                entry_id = command.get("entryId")
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError("fork requires entryId")
                target = next((entry for entry in session.entries if entry.id == entry_id), None)
                if target is None:
                    matches = [entry for entry in session.entries if entry.id.startswith(entry_id)]
                    if len(matches) != 1:
                        raise ValueError("Entry ID is unknown or ambiguous")
                    target = matches[0]
                text = get_message_text(target.message.content) if target.message is not None else ""
                self.agent.messages = session.fork(entry_id)
                restore_compaction(self.agent.compactor, self.agent.messages,
                                   session.latest_custom("compaction"))
                response["data"] = {"text": text, "cancelled": False}
            elif kind == "clone":
                if session.leaf_id is None:
                    raise ValueError("Cannot clone session: no current entry selected")
                self.agent.messages = session.fork(None)
                restore_compaction(self.agent.compactor, self.agent.messages,
                                   session.latest_custom("compaction"))
                response["data"] = {"cancelled": False}
            elif kind == "set_session_name":
                name = command["name"].strip() if isinstance(command.get("name"), str) else ""
                if not name:
                    raise ValueError("Session name cannot be empty")
                session.rename(name)
            else:
                output_path = command.get("outputPath")
                if output_path is not None and not isinstance(output_path, str):
                    raise ValueError("outputPath must be a string")
                if output_path is None:
                    if session.session_file is None:
                        raise ValueError("No active session")
                    output_path = os.path.splitext(session.session_file)[0] + ".html"
                response["data"] = {"path": session.export(output_path)}
        elif kind == "get_fork_messages":
            session = self._require_session()
            messages = []
            for entry in session.get_branch():
                if entry.message is not None and getattr(entry.message, "role", None) == "user":
                    messages.append({"entryId": entry.id, "text": get_message_text(entry.message.content)})
            response["data"] = {"messages": messages}
        elif kind == "get_entries":
            session = self._require_session()
            entries = session.entries
            since = command.get("since")
            if since is not None:
                index = next((i for i, entry in enumerate(entries) if entry.id == since), -1)
                if index == -1:
                    raise ValueError(f"Entry not found: {since}")
                entries = entries[index + 1:]
            response["data"] = {"entries": [entry_to_ts(entry) for entry in entries],
                                "leafId": session.leaf_id}
        elif kind == "get_tree":
            session = self._require_session()
            response["data"] = {"tree": tree_nodes_to_ts(session.entries), "leafId": session.leaf_id}
        else:
            raise ValueError(f"Unknown command: {kind}")
        return False

    async def serve(self, source=None):
        source = source or sys.stdin.buffer
        try:
            while True:
                line = await asyncio.to_thread(source.readline)
                if not line:
                    break
                try:
                    command = json.loads(line)
                    if not isinstance(command, dict):
                        raise ValueError("Command must be a JSON object")
                except (ValueError, UnicodeError) as exc:
                    await self.writer.write({"type": "response", "command": "parse", "success": False,
                                             "error": f"Failed to parse command: {exc}"})
                    continue
                if command.get("type") == "extension_ui_response":
                    self.resolve_ui(command)
                    continue
                await self.dispatch(command)
        finally:
            if self.busy():
                self.agent.abort()
                try:
                    await asyncio.wait_for(self.task, 5)
                except asyncio.TimeoutError:
                    pass
