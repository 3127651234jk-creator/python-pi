"""pi.telemetry: backend-neutral telemetry contracts.

Python port of ``packages/telemetry``: the TelemetryContext/Span contract, a
no-op backend, an in-memory recording backend, and schema definition helpers.
"""

from .memory import InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan
from .noop import NOOP_TELEMETRY_CONTEXT
from .otlp import OTLPTelemetryContext, telemetry_from_env
from .schema import create_typed_span_starter, define_telemetry_schema
from .types import (
    AttributeValue,
    SpanAttributes,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)

__all__ = [
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanStatus",
    "TelemetryContext",
    "TelemetrySpan",
    "NOOP_TELEMETRY_CONTEXT",
    "InMemoryTelemetryContext",
    "OTLPTelemetryContext",
    "telemetry_from_env",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "define_telemetry_schema",
    "create_typed_span_starter",
]
