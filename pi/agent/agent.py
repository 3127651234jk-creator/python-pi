"""Stateful Agent wrapper.

Python port of the core of ``packages/agent/src/agent.ts``: transcript state,
steering and follow-up queues, event subscription, prompt/abort lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, AsyncIterator, Callable, Deque, List, Optional

from ..ai.models import Model
from ..ai.providers import get_default_stream_fn
from ..ai.retry import RetryPolicy, RetrySleepAbortError, is_retryable_assistant_error, retry_delay_ms, sleep_ms
from ..ai.types import SystemMessage, TextContent, ToolResultMessage, UserMessage
from ..util import AbortSignal
from .agent_loop import default_convert_to_llm, run_agent_loop
from .compaction import CompactionSettings, ContextCompactor, is_context_overflow
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    MessageEndEvent,
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
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        self.model = model
        self.compactor = ContextCompactor(compaction or CompactionSettings())
        self.retry = retry
        # Compaction summarization shares the agent's retry policy (TS wires the
        # same settings.retry budget into every summarization call).
        self.compactor.retry = retry
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
        first_error: Optional[Exception] = None
        for listener in list(self._listeners):
            try:
                result = listener(event)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

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
        attempt = 0
        stream = get_default_stream_fn()
        completed_messages: List[Any] = []
        deferred_listener_error: Optional[Exception] = None

        async def emit(event: AgentEvent) -> None:
            nonlocal deferred_listener_error
            # Record completion before subscribers run: a subscriber may persist
            # the message and then another subscriber may raise.
            if isinstance(event, MessageEndEvent):
                completed_messages.append(event.message)
            is_tool_result_event = (getattr(getattr(event, "message", None), "role", None)
                                    == "toolResult")
            if deferred_listener_error is not None and not is_tool_result_event:
                raise deferred_listener_error
            try:
                await self._emit(event)
            except Exception as exc:
                if not is_tool_result_event:
                    raise
                if deferred_listener_error is None:
                    deferred_listener_error = exc

        all_new: List[Any] = []
        try:
            config = self._build_config()
            # Reads the live attempt count: the same config drives every retry
            # round, and the predicate decides whether a failure is superseded.
            config.should_omit_error = lambda message: self._should_retry(message, attempt)
            transcript = list(self.messages)
            prompts_list = prompts

            async def compacted_stream(model, context, options):
                async for event in self.compactor.stream(model, context, options, stream):
                    yield event

            while True:
                new_messages = await run_agent_loop(
                    prompts=prompts_list,
                    context=AgentContext(messages=list(transcript), tools=list(self.tools)),
                    config=config,
                    emit=emit,
                    signal=self._abort,
                    stream_fn=compacted_stream,
                )
                prompts_list = []
                transcript.extend(new_messages)
                all_new.extend(new_messages)

                # A completed assistant response resets the retry budget (TS
                # resets it on the first non-error assistant message); an aborted
                # response ends a retry cycle unsuccessfully.
                last_assistant = next(
                    (m for m in reversed(new_messages) if getattr(m, "role", None) == "assistant"),
                    None,
                )
                if attempt > 0:
                    if last_assistant is not None and last_assistant.stop_reason == "aborted":
                        await self._emit(
                            AutoRetryEndEvent(success=False, attempt=attempt, final_error="Retry cancelled")
                        )
                        attempt = 0
                    elif any(
                        getattr(m, "role", None) == "assistant" and m.stop_reason not in ("error", "aborted")
                        for m in new_messages
                    ):
                        await self._emit(AutoRetryEndEvent(success=True, attempt=attempt))
                        attempt = 0

                failed = next(
                    (
                        m
                        for m in reversed(new_messages)
                        if getattr(m, "role", None) == "assistant" and m.stop_reason == "error"
                    ),
                    None,
                )
                if failed is None:
                    break
                # Context overflow is handled by compaction, not retry.
                if not self._should_retry(failed, attempt):
                    if attempt > 0:
                        await self._emit(
                            AutoRetryEndEvent(success=False, attempt=attempt, final_error=failed.error_message)
                        )
                    break

                attempt += 1
                delay = retry_delay_ms(self.retry, attempt)
                await self._emit(
                    AutoRetryStartEvent(
                        attempt=attempt,
                        max_attempts=self.retry.max_retries,
                        delay_ms=delay,
                        error_message=failed.error_message or "Unknown error",
                    )
                )
                # The superseded attempt stays in the raw transcript (and in
                # session history) but is omitted from model projections, so
                # resume and live runs agree on both the history and the request.
                failed.retried = True
                try:
                    await sleep_ms(delay, self._abort)
                except RetrySleepAbortError:
                    await self._emit(
                        AutoRetryEndEvent(success=False, attempt=attempt, final_error="Retry cancelled")
                    )
                    break
        finally:
            self.is_streaming = False
            self._abort = None
            self.messages = _sync_messages(self, completed_messages)
        return all_new

    def _should_retry(self, message: Any, attempt: int) -> bool:
        """Whether a failed assistant message is retried at this attempt count.

        Shared by the loop and the pre-persist marking callback so the session
        log and the projection agree on which attempts were superseded.
        """
        return (
            self.retry is not None
            and self.retry.enabled
            and attempt < self.retry.max_retries
            and not is_context_overflow(message)
            and is_retryable_assistant_error(message)
        )

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

        async def listener(event: AgentEvent) -> None:
            queue.append(event)

        unsubscribe = self.subscribe(listener)
        task = None
        try:
            task = asyncio.create_task(self.prompt(text))
            while True:
                if queue:
                    yield queue.popleft()
                    continue
                # Only a finished prompt ends the stream: an auto-retry emits its
                # own agent_end for the superseded attempt before continuing.
                if task.done():
                    break
                await asyncio.sleep(0.01)
        finally:
            unsubscribe()
            if task is not None:
                if not task.done():
                    self.abort()
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                else:
                    await task


def _user_input(text: str, images: Optional[List[Any]]) -> UserMessage:
    """One queued user message: plain text, or text plus image blocks."""
    if images:
        content: Any = ([TextContent(text=text)] if text else []) + list(images)
        return UserMessage(content=content, timestamp=_now_ms())
    return UserMessage(content=text, timestamp=_now_ms())


def _sync_messages(agent: "Agent", new_messages: List[Any]) -> List[Any]:
    """Rebuild the transcript from agent state plus the new messages.

    Keep completed messages even when the loop raises before returning its
    result. Completion is recorded before notifying external subscribers.
    """
    return list(agent.messages) + list(new_messages)


def _now_ms() -> int:
    return int(time.time() * 1000)
