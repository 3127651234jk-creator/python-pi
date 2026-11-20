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
from ..ai.types import DoneEvent, SystemMessage, TextContent, ToolResultMessage, UserMessage
from ..util import AbortSignal
from .agent_loop import default_convert_to_llm, run_agent_loop
from .compaction import CompactionSettings, ContextCompactor, is_context_overflow
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    MessageEndEvent,
)


QUEUE_MODES = ("all", "one-at-a-time")


class Agent:
    """Owns the transcript, tool set, and message queues; runs the agent loop."""

    def __init__(
        self,
        model: Model,
        tools: Optional[List[AgentTool]] = None,
        system_prompt: Optional[str] = None,
        thinking_level: str = "off",
        steering_queue_mode: str = "one-at-a-time",
        follow_up_queue_mode: str = "one-at-a-time",
        compaction: Optional[CompactionSettings] = None,
        cache_retention: Optional[str] = "5m",
        block_images: bool = False,
        retry: Optional[RetryPolicy] = None,
        finish_turn: Optional[Callable[..., Any]] = None,
        on_provider_stream_event: Optional[Callable[[Any], Any]] = None,
        queue_mode: Optional[str] = None,
    ) -> None:
        if steering_queue_mode not in QUEUE_MODES or follow_up_queue_mode not in QUEUE_MODES:
            raise ValueError("queue mode must be all or one-at-a-time")
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
        # Each queue drains independently (TS steeringMode/followUpMode); both
        # default to one-at-a-time like TS 1.1.0's queues.
        self.steering_queue_mode = steering_queue_mode
        self.follow_up_queue_mode = follow_up_queue_mode
        if queue_mode is not None:
            # Deprecated alias: one mode for both queues.
            self.steering_queue_mode = queue_mode
            self.follow_up_queue_mode = queue_mode
        self.is_streaming = False
        self.error_message: Optional[str] = None
        # Whether the run that just settled was aborted (port of the TS
        # _agentRunAbortRequested flag reported by agent_settled.aborted).
        self.abort_requested = False
        self._steering_queue: Deque[Any] = deque()
        self._follow_up_queue: Deque[Any] = deque()
        self._listeners: List[Callable[[AgentEvent], Any]] = []
        self._abort: Optional[AbortSignal] = None
        # Per-run signal for abort_retry(): cancels a pending backoff without
        # aborting the run itself (the failed attempt settles the run).
        self._retry_abort: Optional[AbortSignal] = None
        self._run_task: Optional[Any] = None
        self.system_prompt = system_prompt
        self.before_tool_call = None
        self.after_tool_call = None
        self.prepare_request = None
        # Turn-end decision hook ({action: "end"} | {action: "continue"}).
        self.finish_turn = finish_turn
        # Observes raw provider events after parsing (TS onProviderStreamEvent).
        self.on_provider_stream_event = on_provider_stream_event
        # Fired after any queue change (enqueue or drain); may be async.
        # Hosts (RPC mode) use it to broadcast queue_update on consumption.
        self.on_queue_update: Optional[Callable[[], Any]] = None
        # Called with (model, context, options) of the last real provider
        # request that produced an assistant response (cache warming capture).
        self.on_real_request: Optional[Callable[[Any, Any, Any], None]] = None
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

    @property
    def queue_mode(self) -> str:
        """Deprecated combined queue mode: reading reflects steering, writing
        sets both queues. Use ``steering_queue_mode``/``follow_up_queue_mode``."""

        return self.steering_queue_mode

    @queue_mode.setter
    def queue_mode(self, mode: str) -> None:
        if mode not in QUEUE_MODES:
            raise ValueError("queue mode must be all or one-at-a-time")
        self.steering_queue_mode = mode
        self.follow_up_queue_mode = mode

    def _peek(self, queue: Deque[Any], mode: str) -> List[Any]:
        """Next batch a drain would deliver, without consuming (TS queue.peek)."""
        if not queue:
            return []
        if mode == "one-at-a-time":
            return [queue[0]]
        return list(queue)

    def peek_queued_messages(self) -> List[Any]:
        """Preview the messages selected for the next turn without consuming
        them (port of Agent.peekQueuedMessages): the steering batch, or the
        follow-up batch when steering is empty."""

        steering = self._peek(self._steering_queue, self.steering_queue_mode)
        return steering if steering else self._peek(self._follow_up_queue, self.follow_up_queue_mode)

    async def _get_steering_messages(self) -> List[Any]:
        return await self._drain_notified(self._steering_queue, self.steering_queue_mode)

    async def _get_follow_up_messages(self) -> List[Any]:
        return await self._drain_notified(self._follow_up_queue, self.follow_up_queue_mode)

    def _drain(self, queue: Deque[Any], mode: str) -> List[Any]:
        if not queue:
            return []
        if mode == "one-at-a-time":
            return [queue.popleft()]
        messages = list(queue)
        queue.clear()
        return messages

    async def _drain_notified(self, queue: Deque[Any], mode: str) -> List[Any]:
        """Drain a queue and fire the queue-change hook (port of _emitQueueUpdate)."""
        messages = self._drain(queue, mode)
        if messages and self.on_queue_update is not None:
            result = self.on_queue_update()
            if hasattr(result, "__await__"):
                await result
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

    def abort_retry(self) -> None:
        """Cancel a pending auto-retry backoff; the run settles on the failed
        attempt instead of continuing (port of TS AgentSession.abortRetry)."""
        if self._retry_abort is not None:
            self._retry_abort.abort()

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
        self.abort_requested = False
        self._abort = AbortSignal()
        self._retry_abort = AbortSignal()
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
            if isinstance(event, AgentEndEvent):
                # TS AgentSession computes willRetry when agent_end is emitted:
                # no abort requested, retry budget remains, and the final
                # assistant message is a retryable (non-overflow) error.
                failed = next(
                    (m for m in reversed(event.messages) if getattr(m, "role", None) == "assistant"),
                    None,
                )
                event.will_retry = bool(
                    self._abort is not None
                    and not self._abort.aborted
                    and failed is not None
                    and self._should_retry(failed, attempt)
                )
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
            # Compaction summarization reports through the same subscriber
            # pipeline while the run is active.
            self.compactor.event_sink = emit
            transcript = list(self.messages)
            prompts_list = prompts

            def capture_stream(inner):
                """Record the latest request that produced an assistant response:
                the innermost provider boundary, so summarization calls (which
                precede the real request after compaction) are superseded."""
                sink = self.on_real_request

                async def wrapped(model, context, options):
                    async for event in inner(model, context, options):
                        if (sink is not None and isinstance(event, DoneEvent)
                                and getattr(event.message, "role", None) == "assistant"):
                            sink(model, context, options)
                        yield event

                return wrapped

            final_stream = capture_stream(stream) if self.on_real_request is not None else stream

            async def compacted_stream(model, context, options):
                async for event in self.compactor.stream(model, context, options, final_stream):
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
                    await self._sleep_retry_backoff(delay)
                except RetrySleepAbortError:
                    await self._emit(
                        AutoRetryEndEvent(success=False, attempt=attempt, final_error="Retry cancelled")
                    )
                    break
        finally:
            self.is_streaming = False
            self.abort_requested = self._abort is not None and self._abort.aborted
            self._abort = None
            self._retry_abort = None
            self.compactor.event_sink = None
            self.messages = _sync_messages(self, completed_messages)
        return all_new

    async def _sleep_retry_backoff(self, delay: int) -> None:
        """Backoff sleep that ends early on a run abort or an abort_retry call."""
        retry_signal = self._retry_abort
        if retry_signal is None:
            await sleep_ms(delay, self._abort)
            return
        sleep_task = asyncio.ensure_future(sleep_ms(delay, self._abort))
        cancel_task = asyncio.ensure_future(retry_signal.wait())
        try:
            await asyncio.wait({sleep_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task.done() and not (self._abort is not None and self._abort.aborted):
                raise RetrySleepAbortError()
            # A completed sleep (or its abort) raises through here.
            await sleep_task
        finally:
            cancel_task.cancel()
            if not sleep_task.done():
                sleep_task.cancel()

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
            finish_turn=self.finish_turn,
            cache_retention=self.cache_retention,
            on_provider_stream_event=self.on_provider_stream_event,
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
