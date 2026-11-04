"""Telemetry types.

Python port of ``packages/telemetry/src/index.ts`` (runtime surface; the TS
module's type-level schema inference has no Python equivalent and the schema
helpers here are runtime-neutral identity functions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

AttributeValue = Union[str, int, float, bool, Sequence[str], Sequence[int], Sequence[float], Sequence[bool]]
SpanAttributes = Dict[str, Optional[AttributeValue]]


@dataclass
class SpanOptions:
    name: str
    attributes: Optional[SpanAttributes] = None


@dataclass
class SpanError:
    name: str
    message: str


@dataclass
class SpanStatus:
    status: str  # "ok" | "error"
    error: Optional[SpanError] = None


SpanCallback = Callable[["TelemetrySpan"], Union[Any, Awaitable[Any]]]


class TelemetrySpan:
    """One in-progress span; also a parent context for child spans."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        raise NotImplementedError

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        raise NotImplementedError

    def set_attributes(self, attributes: SpanAttributes) -> None:
        raise NotImplementedError

    def set_status(self, status: SpanStatus) -> None:
        raise NotImplementedError


class TelemetryContext:
    """Entry point for starting spans."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        raise NotImplementedError
