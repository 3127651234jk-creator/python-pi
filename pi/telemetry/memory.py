"""In-memory recording telemetry backend.

Python port of ``packages/telemetry/src/memory.ts``: records spans in process
memory with automatic error status when the callback raises. ``start_span`` is
async (the Python equivalent of TS returning a Promise). Create a fresh
instance to isolate tests or independent recording scopes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .noop import NOOP_TELEMETRY_CONTEXT
from .types import SpanCallback, SpanError, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan


@dataclass
class RecordedTelemetryEvent:
    name: str
    attributes: Dict[str, Any]


@dataclass
class RecordedTelemetrySpan:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: Dict[str, Any]
    events: List[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=lambda: SpanStatus(status="ok"))
    settled: bool = False
    end_sequence: Optional[int] = None


@dataclass
class _MutableSpan:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: Dict[str, Any]
    events: List[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=lambda: SpanStatus(status="ok"))
    explicit_status: bool = False
    settled: bool = False
    end_sequence: Optional[int] = None


def _copy_value(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _copy_attributes(attributes):
    copy: Dict[str, Any] = {}
    for name, value in (attributes or {}).items():
        if value is not None:
            copy[name] = _copy_value(value)
    return copy


def _merge_attributes(current, attributes):
    merged = _copy_attributes(current)
    merged.update({k: _copy_value(v) for k, v in (attributes or {}).items() if v is not None})
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.status == "ok":
        return SpanStatus(status="ok")
    if status.error is not None:
        return SpanStatus(status="error", error=SpanError(name=status.error.name, message=status.error.message))
    return SpanStatus(status="error")


async def _start_in_memory_span(state, parent, options: SpanOptions, callback: SpanCallback):
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    span = _MutableSpan(
        id=state["next_span_id"],
        parent_id=parent.id if parent is not None else None,
        name=options.name,
        attributes=_copy_attributes(options.attributes),
    )
    state["next_span_id"] += 1
    state["spans"].append(span)

    def settle(failed: bool, error: object = None) -> None:
        if span.settled:
            return
        if failed and not span.explicit_status:
            if isinstance(error, BaseException):
                span.status = SpanStatus(
                    status="error", error=SpanError(name=type(error).__name__, message=str(error))
                )
            else:
                span.status = SpanStatus(status="error")
        span.settled = True
        span.end_sequence = state["next_end_sequence"]
        state["next_end_sequence"] += 1

    handle = _MemorySpanHandle(state, span)
    try:
        result = callback(handle)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
    except BaseException as exc:  # noqa: BLE001 - telemetry settles on any failure
        settle(True, exc)
        raise
    settle(False)
    return result


class _MemorySpanHandle(TelemetrySpan):
    def __init__(self, state, span: _MutableSpan) -> None:
        self._state = state
        self._span = span

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return _start_in_memory_span(self._state, self._span, options, callback)

    def add_event(self, name: str, attributes=None) -> None:
        if self._span.settled:
            return
        try:
            self._span.events.append(RecordedTelemetryEvent(name=name, attributes=_copy_attributes(attributes)))
        except Exception:
            pass

    def set_attributes(self, attributes) -> None:
        if self._span.settled:
            return
        try:
            self._span.attributes = _merge_attributes(self._span.attributes, attributes)
        except Exception:
            pass

    def set_status(self, status: SpanStatus) -> None:
        if self._span.settled:
            return
        try:
            self._span.status = _copy_status(status)
            self._span.explicit_status = True
        except Exception:
            pass


class InMemoryTelemetryContext(TelemetryContext):
    """Backend-neutral reference implementation that records spans in memory."""

    def __init__(self) -> None:
        self._state = {"spans": [], "next_span_id": 1, "next_end_sequence": 1}

    async def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return await _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> List[RecordedTelemetrySpan]:
        """Return detached snapshots in span-start order."""

        return [
            RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_copy_attributes(span.attributes),
                events=[
                    RecordedTelemetryEvent(name=e.name, attributes=_copy_attributes(e.attributes))
                    for e in span.events
                ],
                status=_copy_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            for span in self._state["spans"]
        ]
