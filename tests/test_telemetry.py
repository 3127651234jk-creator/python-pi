"""Telemetry conformance tests (port of testing/conformance.ts coverage).

Runs the same assertions against any TelemetryContext backend. The in-memory
backend is exercised here; the noop backend only needs to pass through results.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    InMemoryTelemetryContext,
    SpanOptions,
    SpanStatus,
)


def test_records_span_with_attributes_and_events():
    async def scenario():
        telemetry = InMemoryTelemetryContext()
        result = await telemetry.start_span(
            SpanOptions(name="root", attributes={"a": 1}),
            lambda span: _child(span, telemetry),
        )
        assert result == "value"
        spans = telemetry.get_spans()
        assert [s.name for s in spans] == ["root", "child"]
        assert spans[0].parent_id is None
        assert spans[1].parent_id == spans[0].id
        assert spans[0].attributes == {"a": 1}
        assert spans[0].events[0].name == "event"
        assert spans[0].events[0].attributes == {"k": "v"}
        assert all(s.settled for s in spans)
        assert spans[0].status.status == "ok"
        # The child settles before its parent, so end sequences order settlement.
        assert spans[1].end_sequence < spans[0].end_sequence

    asyncio.run(scenario())


async def _child(span, telemetry):
    span.add_event("event", {"k": "v"})
    return await span.start_span(SpanOptions(name="child"), lambda child: "value")


def test_error_status_on_exception():
    async def scenario():
        telemetry = InMemoryTelemetryContext()
        with pytest.raises(RuntimeError):
            await telemetry.start_span(SpanOptions(name="boom"), lambda span: (_ for _ in ()).throw(RuntimeError("nope")))
        span = telemetry.get_spans()[0]
        assert span.status.status == "error"
        assert span.status.error.name == "RuntimeError"
        assert span.status.error.message == "nope"

    asyncio.run(scenario())


def test_explicit_status_wins_over_exception():
    async def scenario():
        telemetry = InMemoryTelemetryContext()

        def fail(span):
            span.set_status(SpanStatus(status="error"))
            raise ValueError("late")

        with pytest.raises(ValueError):
            await telemetry.start_span(SpanOptions(name="s"), fail)
        span = telemetry.get_spans()[0]
        assert span.status.status == "error"
        # Explicit status carries no automatic error details.
        assert span.status.error is None

    asyncio.run(scenario())


def test_noop_after_settle():
    async def scenario():
        telemetry = InMemoryTelemetryContext()
        holder = {}

        async def body(span):
            holder["span"] = span
            return await span.start_span(SpanOptions(name="inner"), lambda s: "x")

        await telemetry.start_span(SpanOptions(name="outer"), body)
        # Mutations after the outer span settled are ignored.
        holder["span"].add_event("late", {"k": 1})
        holder["span"].set_attributes({"late": 2})
        holder["span"].set_status(SpanStatus(status="error"))
        span = telemetry.get_spans()[0]
        assert span.events == []
        assert span.attributes == {}
        assert span.status.status == "ok"
        # The child ran inside the live body, so it was recorded.
        assert len(telemetry.get_spans()) == 2

    asyncio.run(scenario())


def test_noop_backend_passes_values_through():
    async def scenario():
        result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="x"), lambda span: 42)
        assert result == 42

    asyncio.run(scenario())


def test_noop_backend_awaits_async_callbacks():
    """The noop backend must run async span bodies, not return their coroutines."""

    async def scenario():
        ran = []

        async def body(span):
            ran.append(True)
            span.add_event("step")
            return "done"

        result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="x"), body)
        assert result == "done"
        assert ran == [True]

    asyncio.run(scenario())
