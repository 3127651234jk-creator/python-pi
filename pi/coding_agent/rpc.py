"""LF-delimited JSON event output and asynchronous subprocess RPC server.

Commands follow the TypeScript ``pi --mode rpc`` shape: ``{id?, type, ...}``
requests answered by ``{id?, type:"response", command, success, data|error}``,
agent activity streamed as events, plus the extension UI protocol
(``extension_ui_request`` on stdout, ``extension_ui_response`` on stdin).
"""

import asyncio
import json
import sys
import uuid
from typing import Optional

from ..ai.json_io import to_json
from ..ai.types import AssistantMessage, SystemMessage, TextContent
from ..ai.models import ModelError
from ..agent.agent import Agent
from ..agent.compaction import compact_agent, restore_compaction
from .catalog_sync import fetch_catalog, save_cache
from .session import SessionManager
from .model_registry import ModelRegistry
from .resources import ResourceCatalog
from .extensions import ExtensionManager
from .usage import format_usage, session_stats, usage_cost_breakdown


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


def public_model(model):
    return {"provider": model.provider, "id": model.id, "name": model.name,
            "api": model.api, "context_window": model.context_window,
            "max_tokens": model.max_tokens, "reasoning": model.reasoning, "input": model.input}


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
            await self.writer.write({"type": "extension_ui_request", "id": request_id, **request})
            reply = await future
            return reply.get("value") if isinstance(reply, dict) else None
        finally:
            self.ui_futures.pop(request_id, None)

    def resolve_ui(self, reply: dict) -> None:
        future = self.ui_futures.get(reply.get("id"))
        if future is not None and not future.done():
            future.set_result(reply)

    async def run_prompt(self, text):
        async def event(event):
            if self.session and event.type == "message_end":
                self.session.append(event.message)
            await self.writer(event)
        unsubscribe = self.agent.subscribe(event)
        try:
            await self.agent.prompt(text)
        except Exception as exc:
            await self.writer.write({"type": "run_error", "error": str(exc)})
        finally:
            unsubscribe()
            await self.writer.write({"type": "agent_settled"})

    def busy(self):
        return self.task is not None and not self.task.done()

    async def dispatch(self, command):
        response = {"type": "response", "command": command.get("type", "parse"), "success": True}
        if "id" in command:
            response["id"] = command["id"]
        try:
            if "id" in command and not isinstance(command["id"], str):
                raise ValueError("id must be a string")
            kind = command.get("type")
            if not isinstance(kind, str):
                raise ValueError("type must be a string")
            data = None
            if kind in ("prompt", "steer", "follow_up"):
                text = command.get("message")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("message must be a nonempty string")
                if self.extensions and kind == "prompt":
                    if self.busy():
                        raise ValueError("Agent is busy")
                    text = await self.extensions.process_input(text)
                    if text is None:
                        await self.writer.write({**response, "data": {"disposition": "handled"}})
                        return
                if self.resources:
                    text = self.resources.expand(text)
                if kind == "prompt":
                    if self.busy():
                        raise ValueError("Agent is busy; use steer or follow_up")
                    await self.writer.write({**response, "data": {"disposition": "started"}})
                    self.task = asyncio.create_task(self.run_prompt(text))
                    return
                if not self.busy():
                    raise ValueError("Agent is idle; use prompt")
                (self.agent.steer if kind == "steer" else self.agent.follow_up)(text)
            elif kind == "abort":
                self.agent.abort()
            elif kind == "clear_queue":
                self.agent._steering_queue.clear()
                self.agent._follow_up_queue.clear()
            elif kind == "set_steering_mode" or kind == "set_follow_up_mode":
                mode = command["mode"]
                if mode not in ("all", "one-at-a-time"):
                    raise ValueError("mode must be all or one-at-a-time")
                self.agent.queue_mode = mode
            elif kind == "get_session_stats":
                data = session_stats(self.agent.messages, self.agent.model,
                                     self.session.session_file if self.session else None)
            elif kind == "get_usage":
                totals = usage_cost_breakdown(self.agent.messages, self.registry)
                data = {"requests": totals.requests, "input": totals.input, "output": totals.output,
                        "cache_read": totals.cache_read, "cache_write": totals.cache_write,
                        "cost": totals.cost,
                        "by_model": {key: {"requests": entry.requests, "input": entry.input,
                                           "output": entry.output, "cache_read": entry.cache_read,
                                           "cache_write": entry.cache_write, "cost": entry.cost}
                                     for key, entry in totals.by_model.items()}}
            elif kind == "get_last_assistant_text":
                texts = [block.text for message in self.agent.messages if isinstance(message, AssistantMessage)
                         for block in message.content if isinstance(block, TextContent)]
                data = {"text": texts[-1] if texts else ""}
            elif kind == "get_available_thinking_levels":
                data = {"levels": ["off", "minimal", "low", "medium", "high", "xhigh", "max"]}
            elif kind == "compact" and self.registry:
                if self.busy():
                    raise ValueError("Cannot compact while the agent is streaming")
                data = await compact_agent(self.agent, self.session, command.get("instructions", ""))
            elif kind == "sync_models" and self.registry:
                providers = await fetch_catalog()
                save_cache(self.registry.config.agent_dir, providers)
                self.registry.reload()
                data = {"providers": sorted(providers)}
            elif kind == "get_state":
                data = {"is_streaming": self.busy(), "model": public_model(self.agent.model),
                        "thinking_level": self.agent.thinking_level, "message_count": len(self.agent.messages),
                        "session_file": self.session.session_file if self.session else None}
            elif kind == "get_messages":
                data = {"messages": self.agent.messages}
            elif kind == "get_tools":
                data = {"tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.agent.tools]}
            elif kind == "get_commands":
                data = {"commands": [{"name": name, "description": description} for name, (_, description) in self.extensions.commands.items()]} if self.extensions else {"commands": []}
            elif kind == "get_available_models" and self.registry:
                data = {"models": [public_model(m) for m in self.registry.models.values()]}
            else:
                if self.busy():
                    raise ValueError("Command requires an idle agent")
                if kind == "reload_extensions" and self.extensions:
                    await self.extensions.reload()
                elif kind == "set_model" and self.registry:
                    self.agent.model = self.registry.resolve(command.get("provider", self.agent.model.provider), command["model"])
                    data = public_model(self.agent.model)
                elif kind == "set_thinking_level":
                    level = command["level"]
                    if level not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
                        raise ValueError("Invalid thinking level")
                    self.agent.thinking_level = level
                elif kind == "new_session":
                    self.agent.messages = [m for m in self.agent.messages if isinstance(m, SystemMessage)]
                    if self.session:
                        self.session.start_new()
                        for message in self.agent.messages:
                            self.session.append(message)
                    restore_compaction(self.agent.compactor, self.agent.messages,
                                       self.session.latest_custom("compaction") if self.session else None)
                elif kind in ("switch_session", "fork", "clone", "get_tree", "get_entries", "set_session_name", "export") and self.session:
                    if kind == "switch_session":
                        self.agent.messages = self.session.resume(command["path"])
                    elif kind in ("fork", "clone"):
                        self.agent.messages = self.session.fork(command.get("entry_id") if kind == "fork" else None)
                    elif kind == "get_tree":
                        data = {"tree": self.session.tree(), "leaf_id": self.session.leaf_id}
                    elif kind == "get_entries":
                        data = {"entries": self.session.entries}
                    elif kind == "set_session_name":
                        self.session.rename(command["name"])
                    else:
                        data = {"path": self.session.export(command["path"])}
                    if kind in ("switch_session", "fork", "clone"):
                        restore_compaction(self.agent.compactor, self.agent.messages,
                                           self.session.latest_custom("compaction"))
                else:
                    raise ValueError(f"Unsupported command: {kind}")
            response["data"] = data
        except (KeyError, ValueError, TypeError, OSError, RuntimeError, ModelError) as exc:
            response.update(success=False, error=str(exc))
        await self.writer.write(response)

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
                    await self.writer.write({"type": "response", "command": "parse", "success": False, "error": str(exc)})
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
