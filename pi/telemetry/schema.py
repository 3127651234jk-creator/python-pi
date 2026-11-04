"""Schema definition helpers.

Python port of the runtime part of ``packages/telemetry/src/index.ts``. The TS
module's type-level schema inference has no Python equivalent; schemas here
are declarative data with no runtime validation, exactly like the TS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class TelemetryAttributeMetadata:
    description: str
    sensitive: bool = False
    cardinality: Optional[str] = None  # "low" | "high"


@dataclass
class TelemetryAttributeDefinition(TelemetryAttributeMetadata):
    type: str = "string"  # string | number | boolean | string[] | number[] | boolean[]
    values: Optional[Any] = None
    examples: Optional[Any] = None


@dataclass
class TelemetryStartAttributeDefinition(TelemetryAttributeDefinition):
    required: bool = False


@dataclass
class TelemetryEventAttributeDefinition(TelemetryAttributeDefinition):
    required: bool = False


@dataclass
class TelemetryEventDefinition:
    description: str
    attributes: Dict[str, TelemetryEventAttributeDefinition]


@dataclass
class TelemetryParentDefinition:
    kind: str  # "any" | "root_or_external" | "spans"
    spans: Optional[Any] = None


@dataclass
class TelemetrySpanDefinition:
    description: str
    parents: TelemetryParentDefinition
    start_attributes: Dict[str, TelemetryStartAttributeDefinition]
    end_attributes: Dict[str, TelemetryAttributeDefinition]
    events: Optional[Dict[str, TelemetryEventDefinition]] = None
    error_when: str = ""


@dataclass
class TelemetrySchemaDefinition:
    version: int
    spans: Dict[str, TelemetrySpanDefinition]


def define_telemetry_schema(schema: TelemetrySchemaDefinition) -> TelemetrySchemaDefinition:
    """Identity helper marking a schema value as telemetry schema data."""
    return schema


def create_typed_span_starter(
    telemetry_context,
    schema: TelemetrySchemaDefinition,
):
    """Bind a parent context to one schema's span vocabulary.

    Like the TS runtime, this performs no schema validation; it only attaches
    the parent context so nested ``start_span`` calls chain parents.
    """

    async def start(name: str, attributes, callback):
        return await telemetry_context.start_span(
            _options(name, attributes), lambda span: callback(span, _make_starter(span, schema))
        )

    def starter(name: str, attributes, callback):
        return start(name, attributes, callback)

    return starter


def _options(name: str, attributes):
    from .types import SpanOptions

    return SpanOptions(name=name, attributes=attributes)


def _make_starter(span, schema: TelemetrySchemaDefinition) -> Callable:
    def child_starter(name: str, attributes, callback):
        return span.start_span(_options(name, attributes), lambda child: callback(child, _make_starter(child, schema)))

    return child_starter
