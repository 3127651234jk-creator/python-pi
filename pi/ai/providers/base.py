"""Provider adapter infrastructure: stream options, SSE parsing, error helpers."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from ..models import Model
from ..types import (
    AssistantMessage,
    DoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    ThinkingLevel,
    TranscriptContext,
    Usage,
)
from ...util import AbortError, AbortSignal


@dataclass
class SimpleStreamOptions:
    """Options shared by all provider adapters (port of SimpleStreamOptions)."""

    signal: Optional[AbortSignal] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None
    telemetry_context: Optional[Any] = None  # pi.telemetry.TelemetryContext


StreamFn = Callable[[Model, TranscriptContext, Optional[SimpleStreamOptions]], AsyncIterator[Any]]

_THINKING_BUDGETS: Dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}


def thinking_budget(level: Optional[ThinkingLevel]) -> Optional[int]:
    if level is None or level == "off":
        return None
    return _THINKING_BUDGETS.get(level, 8192)


def apply_model_params(body: Dict[str, Any], model: Model, options: Optional[SimpleStreamOptions]) -> None:
    """Merge model-specific request parameters (``Model.params``) into a payload.

    Provider-level options that the caller explicitly set (temperature,
    max_tokens, reasoning effort) keep precedence; every other parameter key is
    applied verbatim, letting models.json tune vendor-specific fields
    (``top_k``, ``presence_penalty``, ``frequency_penalty``, ...).
    """

    explicit = set()
    if options is not None:
        if options.temperature is not None:
            explicit.add("temperature")
        if options.max_tokens:
            explicit.add("max_tokens")
    for key, value in (model.params or {}).items():
        if key not in explicit:
            body[key] = value


async def iter_sse(response: httpx.Response) -> AsyncIterator[Tuple[Optional[str], Any]]:
    """Parse an SSE response body into (event_name, parsed_json) pairs."""

    event_name: Optional[str] = None
    data_lines: List[str] = []

    async def dispatch() -> AsyncIterator[Tuple[Optional[str], Any]]:
        if data_lines:
            data = "\n".join(data_lines)
            parsed: Any = None
            try:
                parsed = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                parsed = data
            yield event_name, parsed

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            async for item in dispatch():
                yield item
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # comments (":...") and unknown fields are ignored

    # Flush a final event that is not followed by a blank line.
    async for item in dispatch():
        yield item


def make_partial(model: Model, timestamp: int) -> AssistantMessage:
    from ..types import Usage

    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=timestamp,
    )


async def run_provider_stream(
    model: Model,
    request_factory: Callable[[], Dict[str, Any]],
    endpoint_path: str,
    headers_factory: Callable[[], Dict[str, str]],
    convert_events: Callable[[Optional[str], Any, "ProviderState"], AsyncIterator[Any]],
    options: Optional[SimpleStreamOptions],
    timestamp: int,
) -> AsyncIterator[Any]:
    """Public entry: dispatches through a telemetry span when configured."""
    options = options or SimpleStreamOptions()
    telemetry = options.telemetry_context
    if telemetry is None:
        async for event in _drive_provider_stream(
            model, request_factory, endpoint_path, headers_factory, convert_events, options, timestamp
        ):
            yield event
        return

    # Bridge the inner generator through a queue so the whole request, including
    # stream consumption, runs inside one telemetry span.
    import asyncio as _asyncio

    from ...telemetry.types import SpanOptions

    queue: "_asyncio.Queue[Any]" = _asyncio.Queue()

    async def body(span):
        try:
            async for event in _drive_provider_stream(
                model, request_factory, endpoint_path, headers_factory, convert_events, options, timestamp
            ):
                if event.type in ("start", "done", "error"):
                    span.add_event(
                        "stream.event", {"event.type": event.type, "stop.reason": getattr(event, "reason", None)}
                    )
                queue.put_nowait(event)
        finally:
            queue.put_nowait(None)

    task = _asyncio.ensure_future(
        telemetry.start_span(
            SpanOptions(
                name="pi.ai.stream",
                attributes={"provider": model.provider, "model": model.id, "api": model.api},
            ),
            body,
        )
    )
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await task
    except BaseException:
        task.cancel()
        raise


async def _drive_provider_stream(
    model: Model,
    request_factory: Callable[[], Dict[str, Any]],
    endpoint_path: str,
    headers_factory: Callable[[], Dict[str, str]],
    convert_events: Callable[[Optional[str], Any, "ProviderState"], AsyncIterator[Any]],
    options: Optional[SimpleStreamOptions],
    timestamp: int,
) -> AsyncIterator[Any]:
    """Shared HTTP/SSE driver for streaming providers.

    ``convert_events`` maps one provider SSE event to zero or more pi events.
    It receives a mutable ``ProviderState`` for the in-progress partial message.
    """

    from ..types import DoneEvent, StreamErrorEvent, StreamStartEvent

    options = options or SimpleStreamOptions()
    signal = options.signal
    state = ProviderState(model=model, timestamp=timestamp)
    started = False
    stop_reason: str = "error"
    error_message: Optional[str] = None
    done_emitted = False
    response: Optional[httpx.Response] = None

    try:
        if signal is not None:
            signal.throw_if_aborted()
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            req = client.build_request(
                "POST",
                model.base_url + endpoint_path,
                json=request_factory(),
                headers=headers_factory(),
            )
            response = await client.send(req, stream=True)
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error_message = f"HTTP {response.status_code} from {model.provider}: {body[:2000]}"
                stop_reason = "error"
            else:
                started = True
                yield StreamStartEvent(partial=state.snapshot())
                async for event_name, data in iter_sse(response):
                    if signal is not None and signal.aborted:
                        raise AbortError()
                    async for converted in convert_events(event_name, data, state):
                        if isinstance(converted, DoneEvent):
                            stop_reason = converted.reason
                            done_emitted = True
                            yield converted
                        elif isinstance(converted, StreamErrorEvent):
                            stop_reason = converted.reason
                            error_message = converted.error.error_message
                            yield converted
                        else:
                            yield converted
                    if done_emitted:
                        break
    except AbortError:
        stop_reason = "aborted"
        error_message = "Request aborted"
    except httpx.HTTPError as exc:
        if not started:
            stop_reason = "error"
            error_message = f"Network error talking to {model.provider}: {exc}"
        else:
            stop_reason = "error"
            error_message = f"Stream error from {model.provider}: {exc}"
    except Exception as exc:  # noqa: BLE001 - encoded as stream error, like the TS contract
        stop_reason = "error"
        error_message = f"Provider error: {exc.__class__.__name__}: {exc}"
    finally:
        if response is not None:
            await response.aclose()

    if stop_reason in ("stop", "length", "toolUse", "deferred") and not done_emitted:
        final = state.snapshot()
        final.stop_reason = stop_reason  # type: ignore[assignment]
        _apply_cost(final, model)
        yield DoneEvent(reason=stop_reason, message=final)  # type: ignore[arg-type]
    elif not done_emitted and not started:
        error = make_partial(model, timestamp)
        error.stop_reason = stop_reason  # type: ignore[assignment]
        error.error_message = error_message or "Unknown provider error"
        yield StreamErrorEvent(reason=stop_reason, error=error)  # type: ignore[arg-type]
    elif not done_emitted:
        error = state.snapshot()
        error.stop_reason = stop_reason  # type: ignore[assignment]
        error.error_message = error_message or "Unknown provider error"
        yield StreamErrorEvent(reason=stop_reason, error=error)  # type: ignore[arg-type]


def _terminated(reason: str) -> bool:
    return reason in ("stop", "length", "toolUse", "deferred")


def _apply_cost(message: AssistantMessage, model: Model) -> None:
    message.usage.cost = model.calculate_cost(message.usage)


class ProviderState:
    """Mutable in-progress assistant message shared by a provider stream."""

    def __init__(self, model: Model, timestamp: int) -> None:
        self._partial = make_partial(model, int(time.time() * 1000))

    @property
    def partial(self) -> AssistantMessage:
        return self._partial

    def snapshot(self) -> AssistantMessage:
        return copy.deepcopy(self._partial)

    def add_text(self, index: int) -> None:
        from ..types import TextContent

        self._partial.content.insert(index, TextContent(text=""))

    def add_thinking(self, index: int) -> None:
        from ..types import ThinkingContent

        self._partial.content.insert(index, ThinkingContent(thinking=""))

    def add_tool_call(self, index: int, tool_call: Any) -> None:
        self._partial.content.insert(index, tool_call)

    def set_usage(self, usage: Usage) -> None:
        self._partial.usage = usage

    def set_response(self, response_id: Optional[str], response_model: Optional[str]) -> None:
        if response_id:
            self._partial.response_id = response_id
        if response_model:
            self._partial.response_model = response_model

    def set_raw_stop_reason(self, raw: Optional[str]) -> None:
        self._partial.raw_stop_reason = raw

    def set_end_turn(self, end_turn: Optional[bool]) -> None:
        self._partial.end_turn = end_turn


def usage_from_counts(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: Optional[int] = None,
) -> Usage:
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
    )
