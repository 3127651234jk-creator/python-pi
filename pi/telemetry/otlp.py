"""OTLP/HTTP(JSON) telemetry backend.

Implements the pi.telemetry contract on top of OpenTelemetry's OTLP/HTTP JSON
export (``POST {endpoint}/v1/traces``). Completed spans are batched per event
loop and sent with httpx; export failures are recorded but never raised —
telemetry must not break the agent. Each root span starts a trace; nested
spans share the trace id and link through ``parentSpanId``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from .noop import NOOP_TELEMETRY_CONTEXT
from .types import SpanCallback, SpanError, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan

NOOP = NOOP_TELEMETRY_CONTEXT


def _attribute_value(value: Any) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_attribute_value(item) for item in value]}}
    return {"stringValue": str(value)}


def _attributes(attributes: Optional[Dict[str, Any]]) -> List[dict]:
    return [{"key": key, "value": _attribute_value(value)}
            for key, value in (attributes or {}).items() if value is not None]


class _SpanRecord:
    def __init__(self, name: str, trace_id: str, span_id: str, parent_span_id: Optional[str],
                 attributes: Optional[Dict[str, Any]]) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.attributes = dict(attributes or {})
        self.events: List[dict] = []
        self.status: SpanStatus = SpanStatus(status="ok")
        self.explicit_status = False
        self.settled = False
        self.start_ns = time.time_ns()
        self.end_ns = self.start_ns

    def to_json(self) -> dict:
        span: Dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(self.end_ns),
            "attributes": _attributes(self.attributes),
            "status": {"code": 1 if self.status.status == "ok" else 2,
                       **({"message": self.status.error.message} if self.status.error else {})},
        }
        if self.parent_span_id:
            span["parentSpanId"] = self.parent_span_id
        if self.events:
            span["events"] = [{"timeUnixNano": str(event.get("_time_ns", self.end_ns)),
                               "name": event["name"], "attributes": _attributes(event.get("attributes"))}
                              for event in self.events]
        return span


def _copy_attributes(attributes):
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in (attributes or {}).items() if v is not None}


class _OTLPSpanHandle(TelemetrySpan):
    def __init__(self, context: "OTLPTelemetryContext", record: _SpanRecord) -> None:
        self._context = context
        self._record = record

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return _start_otlp_span(self._context, self._record, options, callback)

    def add_event(self, name: str, attributes=None) -> None:
        if self._record.settled:
            return
        event = {"name": name, "attributes": _copy_attributes(attributes), "_time_ns": time.time_ns()}
        self._record.events.append(event)

    def set_attributes(self, attributes) -> None:
        if self._record.settled:
            return
        self._record.attributes.update(_copy_attributes(attributes))

    def set_status(self, status: SpanStatus) -> None:
        if self._record.settled:
            return
        self._record.status = SpanStatus(status=status.status,
            error=SpanError(name=status.error.name, message=status.error.message) if status.error else None)
        self._record.explicit_status = True


async def _start_otlp_span(context: "OTLPTelemetryContext", parent: Optional[_SpanRecord],
                           options: SpanOptions, callback: SpanCallback) -> Any:
    if parent is not None and parent.settled:
        return await NOOP.start_span(options, callback)
    trace_id = parent.trace_id if parent is not None else uuid.uuid4().hex
    record = _SpanRecord(options.name, trace_id, uuid.uuid4().hex[:16],
                         parent.span_id if parent is not None else None, _copy_attributes(options.attributes))
    context._records.append(record)
    handle = _OTLPSpanHandle(context, record)

    def settle(failed: bool, error: object = None) -> None:
        if record.settled:
            return
        if failed and not record.explicit_status:
            if isinstance(error, BaseException):
                record.status = SpanStatus(status="error",
                    error=SpanError(name=type(error).__name__, message=str(error)))
            else:
                record.status = SpanStatus(status="error")
        record.settled = True
        record.end_ns = time.time_ns()
    try:
        result = callback(handle)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
    except BaseException as exc:  # noqa: BLE001 - telemetry settles on any failure
        settle(True, exc)
        context._schedule_export()
        raise
    settle(False)
    if parent is None:
        context._schedule_export()
    return result


class OTLPTelemetryContext(TelemetryContext):
    """TelemetryContext exporting completed spans via OTLP/HTTP JSON."""

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None,
                 resource_attributes: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/") + ("/v1/traces" if not endpoint.rstrip("/").endswith("/v1/traces") else "")
        self.headers = {"content-type": "application/json", **(headers or {})}
        self.resource_attributes = resource_attributes or {"service.name": "pi-python"}
        self.timeout = timeout
        self._records: List[_SpanRecord] = []
        self._export_task: Optional[asyncio.Task] = None
        self.last_error: Optional[str] = None

    async def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return await _start_otlp_span(self, None, options, callback)

    def _schedule_export(self) -> None:
        if self._export_task is None or self._export_task.done():
            self._export_task = asyncio.ensure_future(self._export_soon())

    async def _export_soon(self) -> None:
        await asyncio.sleep(1.0)
        try:
            await self.flush()
        except Exception as exc:  # noqa: BLE001 - exports never break the app
            self.last_error = str(exc)

    async def flush(self) -> None:
        """Send all completed spans that have not been exported yet."""
        if not self._records:
            return
        batch, self._records = self._records, []
        spans = [record.to_json() for record in batch if record.settled]
        batch_unsettled = [record for record in batch if not record.settled]
        self._records = batch_unsettled + self._records
        if not spans:
            return
        body = {"resourceSpans": [{"resource": {"attributes": _attributes(self.resource_attributes)},
                                   "scopeSpans": [{"scope": {"name": "pi-python"}, "spans": spans}]}]}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, json=body, headers=self.headers)
        if response.status_code >= 400:
            self.last_error = f"OTLP export failed (HTTP {response.status_code})"
            raise RuntimeError(self.last_error)

    async def close(self) -> None:
        if self._export_task is not None and not self._export_task.done():
            self._export_task.cancel()
        try:
            await self.flush()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)


def telemetry_from_env(environ=None) -> Optional[OTLPTelemetryContext]:
    """Build an OTLP context from PI_PY_TELEMETRY_ENDPOINT (+ optional JSON headers)."""
    import json as _json
    import os as _os

    environ = environ if environ is not None else _os.environ
    endpoint = environ.get("PI_PY_TELEMETRY_ENDPOINT")
    if not endpoint:
        return None
    headers = {}
    raw_headers = environ.get("PI_PY_TELEMETRY_HEADERS")
    if raw_headers:
        try:
            parsed = _json.loads(raw_headers)
            if isinstance(parsed, dict):
                headers = {str(k): str(v) for k, v in parsed.items()}
        except ValueError:
            pass
    service = environ.get("PI_PY_TELEMETRY_SERVICE", "pi-python")
    return OTLPTelemetryContext(endpoint, headers=headers, resource_attributes={"service.name": service})
