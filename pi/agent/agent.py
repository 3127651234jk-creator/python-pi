"""Stateful Agent wrapper.

Python port of the core of ``packages/agent/src/agent.ts``: transcript state,
steering and follow-up queues, event subscription, prompt/abort lifecycle.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, AsyncIterator, Callable, Deque, List, Optional

from ..ai.models import Model
from ..ai.providers import get_default_stream_fn
from ..ai.types import SystemMessage, TextContent, ToolResultMessage, UserMessage
from ..util import AbortSignal
from .agent_loop import default_convert_to_llm, run_agent_loop
from .compaction import CompactionSettings, ContextCompactor
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
)


class Agent:
    """Owns the transcript, tool set, and message queues; runs the agent loop."""

    def __init__(
        self,
        model: Model,
        tools: Optional[List[AgentTool]] = None,
        system_prompt: Optional[str] = None,
        thinking_level: str = "off",
        queue_mode: str = "all",
        compaction: Optional[CompactionSettings] = None,
        cache_retention: Optional[str] = "5m",
        block_images: bool = False,
    ) -> None:
        self.model = model
        self.compactor = ContextCompactor(compaction or CompactionSettings())
        self.thinking_level = thinking_level
        self.cache_retention = cache_retention if cache_retention in ("5m", "1h") else None
        self.block_images = block_images
        self.tools: List[AgentTool] = list(tools or [])
        self.messages: List[Any] = []
        self.queue_mode = queue_mode
        self.is_streaming = False
        self.error_message: Optional[str] = None
        self._steering_queue: Deque[Any] = deque()
        self._follow_up_queue: Deque[Any] = deque()
        self._listeners: List[Callable[[AgentEvent], Any]] = []
        self._abort: Optional[AbortSignal] = None
        self._run_task: Optional[Any] = None
        self.system_prompt = system_prompt
        self.before_tool_call = None
        self.after_tool_call = None
        self.prepare_request = None
        self._request_model = None
        if system_prompt:
            self.messages.append(SystemMessage(content=system_prompt, timestamp=_now_ms()))

    # -- subscription ------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        """Register an async or sync listener; returns an unsubscribe fn."""

        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def _emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            result = listener(event)
            if hasattr(result, "__await__"):
                await result

    # -- queues ------------------------------------------------------------

    async def _get_steering_messages(self) -> List[Any]:
        return self._drain(self._steering_queue)

    async def _get_follow_up_messages(self) -> List[Any]:
        return self._drain(self._follow_up_queue)

    def _drain(self, queue: Deque[Any]) -> List[Any]:
        if not queue:
            return []
        if self.queue_mode == "one-at-a-time":
            return [queue.popleft()]
        messages = list(queue)
        queue.clear()
        return messages

    def steer(self, text: str, images: Optional[List[Any]] = None) -> None:
        """Queue a message to inject at the next drain point (mid-run)."""
        self._steering_queue.append(_user_input(text, images))

    def follow_up(self, text: str, images: Optional[List[Any]] = None) -> None:
        """Queue a message to process after the agent would otherwise stop."""
        self._follow_up_queue.append(_user_input(text, images))

    # -- lifecycle -----------------------------------------------------------

    def abort(self) -> None:
        if self._abort is not None:
            self._abort.abort()

    async def prompt(
        self,
        text: str,
        system_message: Optional[SystemMessage] = None,
        images: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Add a user prompt (with optional image attachments and system message) and run the loop."""

        prompts: List[Any] = []
        if system_message is not None:
            prompts.append(system_message)
        if images:
            content: Any = [TextContent(text=text)] + list(images) if text else list(images)
            prompts.append(UserMessage(content=content, timestamp=_now_ms()))
        else:
            prompts.append(UserMessage(content=text, timestamp=_now_ms()))
        return await self._run(prompts)

    async def _run(self, prompts: List[Any]) -> List[Any]:
        if self.is_streaming:
            raise RuntimeError("Agent is already streaming")
        self.is_streaming = True
        self.error_message = None
        self._abort = AbortSignal()
        config = self._build_config()
        new_messages: List[Any] = []
        stream = get_default_stream_fn()

        async def compacted_stream(model, context, options):
            async for event in self.compactor.stream(model, context, options, stream):
                yield event

        try:
            new_messages = await run_agent_loop(
                prompts=prompts,
                context=AgentContext(messages=list(self.messages), tools=list(self.tools)),
                config=config,
                emit=self._emit,
                signal=self._abort,
                stream_fn=compacted_stream,
            )
        finally:
            self.is_streaming = False
            self._abort = None
            self.messages = _sync_messages(self, new_messages)
        return new_messages

    def _build_config(self) -> AgentLoopConfig:
        async def prepare_request(context, signal):
            self._request_model = None
            if self.prepare_request is None:
                return None
            update = self.prepare_request(context, signal)
            if hasattr(update, "__await__"):
                update = await update
            if update is not None and update.model is not None:
                self._request_model = update.model
            return update

        async def get_api_key(provider):
            model = self._request_model if self._request_model is not None else self.model
            if model.get_api_key is None:
                return model.api_key
            return await model.get_api_key(provider)

        # Read block_images through the attribute so mid-session setting
        # changes take effect on the next request.
        def convert_to_llm(messages: List[Any]) -> List[Any]:
            return default_convert_to_llm(messages, block_images=self.block_images)

        return AgentLoopConfig(
            model=self.model,
            get_api_key=get_api_key,
            convert_to_llm=convert_to_llm,
            reasoning=None if self.thinking_level == "off" else self.thinking_level,
            get_steering_messages=self._get_steering_messages,
            get_follow_up_messages=self._get_follow_up_messages,
            tool_execution="parallel",
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            prepare_request=prepare_request,
            cache_retention=self.cache_retention,
        )

    async def stream_events(self, text: str) -> AsyncIterator[AgentEvent]:
        """Run a prompt while yielding every event (subscribe-free helper)."""

        queue: Deque[AgentEvent] = deque()
        done = {"flag": False}

        async def listener(event: AgentEvent) -> None:
            queue.append(event)
            if isinstance(event, AgentEndEvent):
                done["flag"] = True

        unsubscribe = self.subscribe(listener)
        task = None
        try:
            task = _create_task(self.prompt(text))
            while True:
                if queue:
                    yield queue.popleft()
                    continue
                if done["flag"] and not queue:
                    break
                await _sleep_small()
        finally:
            unsubscribe()
            if task is not None:
                await task


def _user_input(text: str, images: Optional[List[Any]]) -> UserMessage:
    """One queued user message: plain text, or text plus image blocks."""
    if images:
        content: Any = ([TextContent(text=text)] if text else []) + list(images)
        return UserMessage(content=content, timestamp=_now_ms())
    return UserMessage(content=text, timestamp=_now_ms())


def _create_task(coro: Any) -> Any:
    import asyncio

    return asyncio.ensure_future(coro)


async def _sleep_small() -> None:
    import asyncio

    await asyncio.sleep(0.01)


def _sync_messages(agent: "Agent", new_messages: List[Any]) -> List[Any]:
    """Rebuild the transcript from agent state plus the new messages.

    The loop mutates its own context copy; the Agent keeps one authoritative
    list: everything that was in ``self.messages`` before the run, plus the
    messages appended by the run.
    """
    return list(agent.messages) + list(new_messages)


def _now_ms() -> int:
    return int(time.time() * 1000)
