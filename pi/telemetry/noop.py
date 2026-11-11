"""No-op telemetry backend.

Python port of ``packages/telemetry/src/noop.ts``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .types import SpanCallback, SpanOptions, TelemetryContext, TelemetrySpan


class _NoopSpan(TelemetrySpan):
    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    def add_event(self, name, attributes=None):
        return None

    def set_attributes(self, attributes):
        return None

    def set_status(self, status):
        return None


class _NoopTelemetryContext(TelemetryContext):
    async def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        result = callback(_NoopSpan())
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            return await result
        return result


NOOP_TELEMETRY_CONTEXT = _NoopTelemetryContext()
