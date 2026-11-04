"""Eval framework essentials.

The TypeScript evals build on the ``vitest-evals`` npm package. This module is
the Python port's local replacement for the parts pi's eval system actually
uses: the ``Harness`` adapter contract, run normalization, transcript events,
a structured-output judge, and a runner that executes one eval case and writes
a Vitest-schema JSON report.

Transcript events, usage summaries and harness runs keep the TypeScript
camelCase field names because they are interchange data embedded in
``vitest.json`` reports and read back by :mod:`pi.evals.report`.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..util import AbortSignal

# ---------------------------------------------------------------------------
# JSON helpers (port of vitest-evals toJsonValue / normalizeRecord)
# ---------------------------------------------------------------------------


def to_json_value(value: Any) -> Any:
    """Normalize a value into the JSON-safe shape used by harness runs."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            converted = to_json_value(item)
            if converted is not None or item is None:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return None


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON properties from a record while preserving valid values."""
    return {key: to_json_value(item) for key, item in record.items()
            if to_json_value(item) is not None or item is None}


def serialize_error(error: Any) -> Dict[str, Any]:
    """Serialize an arbitrary thrown value into the normalized error shape."""
    if isinstance(error, dict):
        return to_json_value(error) or {}
    return {"name": type(error).__name__, "message": str(error)}


# ---------------------------------------------------------------------------
# Harness contract (port of vitest-evals createHarness / normalizeHarnessRun)
# ---------------------------------------------------------------------------


@dataclass
class HarnessContext:
    """Runtime context passed from the eval fixture into a harness run."""

    signal: Optional[AbortSignal] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def set_artifact(self, name: str, value: Any) -> None:
        """Store one JSON-safe artifact on the current run."""
        self.artifacts[name] = value


@dataclass
class HarnessRunArgs:
    """Arguments passed to the ``create_harness`` run callback."""

    input: Any
    signal: Optional[AbortSignal] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    set_artifact: Callable[[str, Any], None] = lambda name, value: None


@dataclass
class Harness:
    """Adapter that executes the system under test and returns a run."""

    name: str
    run: Callable[[HarnessRunArgs], Any]  # sync or async; SimpleHarnessResult or HarnessRun


def create_harness(name: str, run: Callable[[HarnessRunArgs], Any]) -> Harness:
    """Create a harness from the common "run app code and return output" shape."""
    return Harness(name=name, run=run)


def is_harness_run(value: Any) -> bool:
    """True when a value already matches the normalized run contract."""
    return (isinstance(value, dict) and isinstance(value.get("session"), dict)
            and isinstance(value["session"].get("events"), list))


def normalize_harness_run(input_value: Any, result: Any, context: Optional[HarnessContext] = None) -> Dict[str, Any]:
    """Normalize a lightweight harness result into the reporter-facing run shape."""
    if is_harness_run(result):
        if context and context.artifacts:
            merged = dict(result)
            merged["artifacts"] = {**merged.get("artifacts", {}), **context.artifacts}
            return merged
        return result
    result = result or {}
    artifacts = {**(result.get("artifacts") or {})}
    if context:
        artifacts.update(context.artifacts)
    return {
        "input": to_json_value(input_value),
        "output": to_json_value(result.get("output")),
        "session": {"events": list(result.get("events") or [])},
        "usage": to_json_value(result.get("usage") or {}),
        "timings": to_json_value(result.get("timings") or {}),
        "artifacts": artifacts,
        "metadata": to_json_value(result.get("metadata") or {}),
        "errors": [serialize_error(error) for error in (result.get("errors") or [])],
    }


class HarnessRunError(Exception):
    """Error carrying a partial or complete normalized harness run."""

    def __init__(self, error: Any, harness_run: Dict[str, Any]) -> None:
        super().__init__(str(error))
        self.harness_run = harness_run


def attach_harness_run_to_error(error: Any, harness_run: Dict[str, Any]) -> HarnessRunError:
    """Attach a partial or complete harness run to an arbitrary thrown error."""
    attached = HarnessRunError(error, harness_run)
    attached.__cause__ = error
    return attached


def get_harness_run_from_error(error: Any) -> Optional[Dict[str, Any]]:
    """Read an attached harness run back off a previously wrapped error value."""
    return error.harness_run if isinstance(error, HarnessRunError) else None


def current_system_prompt(messages: List[Any]) -> str:
    """The system prompt the transcript replays (first system message text)."""
    from ..ai.types import get_system_message_text

    for message in messages:
        if getattr(message, "role", None) == "system":
            return get_system_message_text(message)
    return ""


# ---------------------------------------------------------------------------
# Structured output judge (port of vitest-evals StructuredOutputJudge)
# ---------------------------------------------------------------------------


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _strict_equals(expected: Any, actual: Any) -> bool:
    if expected is actual:
        return True
    if expected is None or actual is None:
        return False
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_strict_equals(e, a) for e, a in zip(expected, actual))
    if _is_record(expected) and _is_record(actual):
        return (set(expected) == set(actual)
                and all(_strict_equals(value, actual[key]) for key, value in expected.items()))
    return False


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _format_mismatch_details(mismatches: List[Dict[str, Any]]) -> str:
    return "; ".join(f"{m['key']}: expected {_format_value(m['expected'])}, got {_format_value(m['actual'])}"
                     for m in mismatches)


@dataclass
class StructuredOutputJudge:
    """Scores a JSON response against expected fields."""

    expected: Dict[str, Any]
    match: Union[str, Callable[[Any, Any, str], bool]] = "strict"
    require_all: bool = True
    allow_extras: bool = True
    error_field: Optional[str] = "error"
    name: str = "StructuredOutputJudge"

    def assess(self, *, output: Any, input: Any = None, expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return ``{"score": float, "metadata": {...}}`` for one response.

        ``input`` is accepted for judge-protocol parity and ignored.
        """
        expected_fields = self.expected if expected is None else expected
        if isinstance(output, str):
            text = output
        elif output is not None:
            try:
                text = json.dumps(output, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(output)
        else:
            text = ""
        try:
            parsed = json.loads(text)
        except ValueError as error:
            return {"score": 0, "metadata": {"rationale": f"Failed to parse output as JSON: {error}", "output": text}}
        if not expected_fields:
            return {"score": 1, "metadata": {"rationale": "Valid JSON output (no expected fields specified)"}}
        if self.error_field is not None:
            error_value = parsed.get(self.error_field) if isinstance(parsed, dict) else None
            if error_value and error_value != "" and error_value is not None:
                return {"score": 0, "metadata": {"rationale": f"Output contains error: {error_value}", "output": text}}
        matches: List[str] = []
        mismatches: List[Dict[str, Any]] = []
        for key, expected_value in expected_fields.items():
            actual_value = parsed.get(key) if isinstance(parsed, dict) else None
            if self._field_matches(expected_value, actual_value, key):
                matches.append(key)
            else:
                mismatches.append({"key": key, "expected": expected_value, "actual": actual_value})
        extras = [key for key in (parsed if isinstance(parsed, dict) else {}) if key not in expected_fields]
        total_expected = len(expected_fields)
        total_matched = len(matches)
        if self.require_all and mismatches:
            return {"score": 0, "metadata": {
                "rationale": (f"Missing required fields: {', '.join(m['key'] for m in mismatches)}"
                              f" - {_format_mismatch_details(mismatches)}"),
            }}
        if not self.allow_extras and extras:
            return {"score": 0, "metadata": {"rationale": f"Unexpected extra fields: {', '.join(extras)}"}}
        score = total_matched / total_expected if total_expected > 0 else 1
        if score == 1:
            extra_info = f" (plus extra fields: {', '.join(extras)})" if extras else ""
            return {"score": 1, "metadata": {"rationale": f"All expected fields match{extra_info}"}}
        return {"score": score, "metadata": {
            "rationale": (f"Matched {total_matched}/{total_expected} fields"
                          f" - {_format_mismatch_details(mismatches)}"),
            "matched": total_matched, "total": total_expected,
        }}

    def _field_matches(self, expected: Any, actual: Any, key: str) -> bool:
        if callable(self.match):
            return bool(self.match(expected, actual, key))
        if self.match == "strict":
            return _strict_equals(expected, actual)
        raise ValueError(f"Unsupported match strategy: {self.match!r}")


def structured_output_judge(
    expected: Dict[str, Any],
    match: Union[str, Callable[[Any, Any, str], bool]] = "strict",
    allow_extras: bool = False,
    require_all: bool = True,
    error_field: Optional[str] = "error",
) -> StructuredOutputJudge:
    """Build a deterministic structured-output judge.

    ``match="strict"`` requires exact recursive equality; ``allow_extras=False``
    rejects responses with fields beyond the expected set.
    """
    return StructuredOutputJudge(
        expected=expected, match=match, require_all=require_all,
        allow_extras=allow_extras, error_field=error_field,
    )


# ---------------------------------------------------------------------------
# Eval case definition and registry (port of describeEval)
# ---------------------------------------------------------------------------


@dataclass
class EvalCaseDef:
    """One registered eval case: an eval set, a title, and its body."""

    eval_set: str
    title: str
    body: Callable[[Callable[[Any], Any]], Any]
    harness: Harness
    judges: List[Any] = field(default_factory=list)
    judge_threshold: Optional[float] = None
    file: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.eval_set} > {self.title}"


_REGISTRY: List[EvalCaseDef] = []


def describe_eval(
    name: str,
    *,
    harness: Harness,
    judges: Optional[List[Any]] = None,
    judge_threshold: Optional[float] = None,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Register one eval case; the decorated function is the case body.

    The body receives an async ``run(input)`` callable that executes the harness
    and resolves to the normalized run. Use ``judge_threshold=None`` for
    comparative scoring: a low score is data, not an infrastructure failure.
    """
    def decorator(body: Callable[[Callable[[Any], Any]], Any]) -> Callable[[Any], Any]:
        _REGISTRY.append(EvalCaseDef(
            eval_set=name,
            title=body.__name__.replace("_", " "),
            body=body,
            harness=harness,
            judges=list(judges or []),
            judge_threshold=judge_threshold,
        ))
        return body

    return decorator


def load_cases_from_file(path: Union[str, Path]) -> List[EvalCaseDef]:
    """Import one eval file and return the cases it registered."""
    before = len(_REGISTRY)
    module_name = f"_pi_eval_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load eval file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = _REGISTRY[before:]
    for case in cases:
        case.file = str(path)
    return cases


# ---------------------------------------------------------------------------
# Runner: one case in, one Vitest-schema report out
# ---------------------------------------------------------------------------


async def _run_case(case: EvalCaseDef) -> Dict[str, Any]:
    """Execute one case and return its assertion record (Vitest JSON shape)."""
    state: Dict[str, Any] = {"run": None, "input": None}

    async def run(input_value: Any) -> Dict[str, Any]:
        state["input"] = input_value
        context = HarnessContext()
        args = HarnessRunArgs(input=input_value, signal=context.signal,
                              artifacts=context.artifacts, set_artifact=context.set_artifact)
        try:
            result = case.harness.run(args)
            if inspect.isawaitable(result):
                result = await result
        except HarnessRunError as error:
            state["run"] = error.harness_run
            raise
        state["run"] = normalize_harness_run(input_value, result, context)
        return state["run"]

    error: Optional[BaseException] = None
    try:
        outcome = case.body(run)
        if inspect.isawaitable(outcome):
            await outcome
        if state["run"] is None:
            raise ValueError("Eval case did not call run().")
    except Exception as exc:  # noqa: BLE001 - eval failures become test failures
        error = exc
    harness_run = state["run"]

    scores: List[Dict[str, Any]] = []
    if error is None and harness_run is not None:
        for judge in case.judges:
            outcome = judge.assess(input=state["input"], output=harness_run.get("output"))
            scores.append({"name": getattr(judge, "name", type(judge).__name__), "score": outcome["score"]})
    avg_score = sum(s["score"] for s in scores) / len(scores) if scores else None
    threshold_failed = (case.judge_threshold is not None and avg_score is not None
                        and avg_score < case.judge_threshold)

    meta: Dict[str, Any] = {"eval": {"avgScore": avg_score, "scores": scores, "thresholdFailed": threshold_failed}}
    if harness_run is not None:
        meta["harness"] = {"name": case.harness.name, "run": harness_run}
    failures: List[str] = []
    if error is not None:
        failures.append(f"{type(error).__name__}: {error}")
    if threshold_failed:
        failures.append(f"judge threshold not met: {avg_score} < {case.judge_threshold}")
    status = "failed" if failures else "passed"
    return {
        "ancestorTitles": [case.eval_set],
        "fullName": case.full_name,
        "status": status,
        "title": case.title,
        "failureMessages": failures,
        "meta": meta,
    }


def case_report(assertion: Dict[str, Any], file: str, start_ms: int) -> Dict[str, Any]:
    """Wrap one assertion into the single-case Vitest JSON report shape."""
    failed = assertion["status"] == "failed"
    return {
        "numFailedTests": 1 if failed else 0,
        "numPassedTests": 0 if failed else 1,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "numTotalTests": 1,
        "startTime": start_ms,
        "success": not failed,
        "testResults": [{
            "message": "",
            "name": file,
            "status": "failed" if failed else "passed",
            "assertionResults": [assertion],
        }],
    }


def discover_eval_files(files: List[str]) -> List[Dict[str, str]]:
    """Import eval files and return ``{"name", "file"}`` discovery records."""
    discovered: List[Dict[str, str]] = []
    for file in files:
        for case in load_cases_from_file(file):
            discovered.append({"name": case.full_name, "file": file})
    return discovered


async def run_eval_file(file: str, test_name_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Run one eval file's selected case and return the Vitest JSON report."""
    started_ms = int(time.time() * 1000)
    cases = load_cases_from_file(file)
    selected = [case for case in cases
                if test_name_pattern is None or re.search(test_name_pattern, case.full_name)]
    if len(selected) != 1:
        raise ValueError(f"Eval selection must match exactly one case in {file}: {test_name_pattern!r}")
    assertion = await _run_case(selected[0])
    return case_report(assertion, file, started_ms)
