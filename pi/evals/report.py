"""Observation reading and paired comparison reporting.

Port of ``packages/evals/src/report.ts``. Observations, comparisons and totals
are JSON-shaped dictionaries (TypedDicts) because they are written to
``observations.jsonl`` and ``report.json`` exactly as the TypeScript runner
does; the field names stay camelCase for interchange compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict

from .plan import EvalTask

PI_SESSION_SNAPSHOT_ARTIFACT = "piSessionJsonl"

CONTROL = "without_docs"
TREATMENT = "with_docs"
VARIANTS = (CONTROL, TREATMENT)


class EvalRunIdentity(TypedDict):
    evalSet: str
    caseId: str
    variant: str
    model: str
    runNumber: int


class EvalMetrics(TypedDict, total=False):
    inputTokens: float
    outputTokens: float
    cacheReadTokens: float
    cacheWriteTokens: float
    totalTokens: float
    toolCalls: float
    totalMs: float
    estimatedCostUsd: float


class EvalObservation(EvalRunIdentity, EvalMetrics, total=False):
    outcome: str  # "scored" | "unscored" | "skipped" | "pending" | "errored"
    score: float


class PairedMetricSummary(TypedDict):
    eligiblePairs: int
    controlMean: Optional[float]
    treatmentMean: Optional[float]
    meanDelta: Optional[float]


class EvalSetComparison(TypedDict):
    evalSet: str
    totalPairs: int
    eligiblePairs: int
    blockedPairs: int
    controlPassRate: Optional[float]
    treatmentPassRate: Optional[float]
    lift: Optional[float]
    flags: List[str]
    totalTokens: PairedMetricSummary
    toolCalls: PairedMetricSummary
    totalMs: PairedMetricSummary
    estimatedCostUsd: PairedMetricSummary


class BlockedPair(TypedDict):
    evalSet: str
    caseId: str
    model: str
    runNumber: int
    reasons: List[str]


class OperationalMetricTotal(TypedDict):
    availableRuns: int
    total: Optional[float]


class VariantTotals(TypedDict):
    variant: str
    runs: int
    inputTokens: OperationalMetricTotal
    outputTokens: OperationalMetricTotal
    cacheReadTokens: OperationalMetricTotal
    cacheWriteTokens: OperationalMetricTotal
    totalTokens: OperationalMetricTotal
    toolCalls: OperationalMetricTotal
    totalMs: OperationalMetricTotal
    estimatedCostUsd: OperationalMetricTotal


class EvalComparisonReport(TypedDict):
    schemaVersion: int
    protocolDigest: str
    control: str
    treatment: str
    comparisons: List[EvalSetComparison]
    blockedPairs: List[BlockedPair]
    operationalTotals: List[VariantTotals]


def _identity(eval_set: str, case_id: str, variant: str, model: str, run_number: int) -> EvalRunIdentity:
    return {"evalSet": eval_set, "caseId": case_id, "variant": variant,
            "model": model, "runNumber": run_number}


def task_identity(task: EvalTask) -> EvalRunIdentity:
    return _identity(task.eval_set, task.case_id, task.variant, task.model, task.run_number)


def _observation(identity: EvalRunIdentity, **extra: Any) -> EvalObservation:
    return {**identity, **extra}  # type: ignore[typeddict-item]


def errored_observation(task: EvalTask) -> EvalObservation:
    return _observation(task_identity(task), outcome="errored")


def optional_metric(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise TypeError(f"{name} must be a finite non-negative number.")
    return value


def validate_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 <= value <= 1):
        raise TypeError("Eval score must be between 0 and 1.")
    return value


def classify_case_status(status: str) -> Optional[str]:
    """Map a Vitest test status to an observation outcome, if it is not passed."""
    if status == "failed":
        return "errored"
    if status in ("skipped", "todo", "disabled"):
        return "skipped"
    if status == "pending":
        return "pending"
    return None


# ---------------------------------------------------------------------------
# Report reading (local replacement for @vitest-evals/core/node)
# ---------------------------------------------------------------------------


def read_vitest_json_report_file(report_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and validate the raw Vitest JSON report."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("testResults"), list):
        raise ValueError("Vitest JSON report must contain testResults.")
    for result in data["testResults"]:
        if not isinstance(result, dict) or not isinstance(result.get("assertionResults"), list):
            raise ValueError("Vitest JSON testResults must contain assertionResults.")
        for assertion in result["assertionResults"]:
            if (not isinstance(assertion, dict) or not isinstance(assertion.get("fullName"), str)
                    or not isinstance(assertion.get("status"), str)):
                raise ValueError("Vitest JSON assertion is invalid.")
    return data


def read_report_workspace(report_paths: Sequence[Union[str, Path]]) -> Dict[str, Any]:
    """Collect scored harness cases from one or more Vitest JSON reports."""
    cases: List[Dict[str, Any]] = []
    for report_path in report_paths:
        report = read_vitest_json_report_file(report_path)
        for result in report["testResults"]:
            for assertion in result["assertionResults"]:
                meta = assertion.get("meta")
                if not isinstance(meta, dict):
                    continue
                cases.append({
                    "fullName": assertion["fullName"],
                    "status": assertion["status"],
                    "eval": meta.get("eval"),
                    "harness": meta.get("harness"),
                })
    return {"cases": cases}


def persist_session_sync(case_result: Mapping[str, Any], task: EvalTask, artifact_directory: str) -> None:
    """Write the run's session snapshot under the variant's sessions directory."""
    harness = case_result.get("harness")
    run = harness.get("run") if isinstance(harness, dict) else None
    artifacts = run.get("artifacts") if isinstance(run, dict) else None
    session = artifacts.get(PI_SESSION_SNAPSHOT_ARTIFACT) if isinstance(artifacts, dict) else None
    if not isinstance(session, str):
        return
    identity = json.dumps([task.eval_set, task.case_id, task.variant, task.model, task.run_number],
                          ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    directory = Path(artifact_directory) / task.variant / "sessions" / digest
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / "session.jsonl"
    path.write_text(session, encoding="utf-8")
    os.chmod(path, 0o600)


async def read_task_observation(task: EvalTask, report_path: str, artifact_directory: str) -> EvalObservation:
    """Read one task's normalized outcome and telemetry from its report."""
    identity = task_identity(task)
    try:
        workspace = read_report_workspace([report_path])
        raw_report = read_vitest_json_report_file(report_path)
    except Exception:  # noqa: BLE001 - unreadable reports are errored observations
        return _observation(identity, outcome="errored")
    reported_full_name = f"{task.eval_set} {task.case_id}"
    assertions = [assertion for result in raw_report["testResults"] for assertion in result["assertionResults"]]
    if len(assertions) != 1:
        return _observation(identity, outcome="errored")
    assertion = assertions[0]
    if assertion["fullName"] != reported_full_name:
        return _observation(identity, outcome="errored")
    status_outcome = classify_case_status(assertion["status"])
    if status_outcome in ("skipped", "pending"):
        return _observation(identity, outcome=status_outcome)
    if len(workspace["cases"]) != 1:
        return _observation(identity, outcome="errored")
    case_result = workspace["cases"][0]
    if case_result["fullName"] != reported_full_name:
        return _observation(identity, outcome="errored")
    if case_result["status"] != assertion["status"]:
        return _observation(identity, outcome="errored")
    run = (case_result.get("harness") or {}).get("run") if isinstance(case_result.get("harness"), dict) else None
    if not isinstance(run, dict):
        return _observation(identity, outcome="errored")
    persist_session_sync(case_result, task, artifact_directory)
    usage = run.get("usage") or {}
    provider = usage.get("provider")
    model = usage.get("model")
    actual_model = f"{provider}/{model}" if provider and model else None
    if actual_model != task.model:
        return _observation(identity, outcome="errored")
    usage_metadata = usage.get("metadata") or {}
    timings = run.get("timings") or {}
    metrics: EvalMetrics = {}
    try:
        for key, value in (
            ("inputTokens", usage.get("inputTokens")),
            ("outputTokens", usage.get("outputTokens")),
            ("cacheReadTokens", usage_metadata.get("cacheReadTokens")),
            ("cacheWriteTokens", usage_metadata.get("cacheWriteTokens")),
            ("totalTokens", usage.get("totalTokens")),
            ("toolCalls", usage.get("toolCalls")),
            ("totalMs", timings.get("totalMs")),
            ("estimatedCostUsd", usage_metadata.get("estimatedCostUsd")),
        ):
            metric = optional_metric(value, key)
            if metric is not None:
                metrics[key] = metric  # type: ignore[typeddict-item]
    except TypeError:
        return _observation(identity, outcome="errored")
    if status_outcome == "errored" or run.get("errors"):
        return _observation(identity, **metrics, outcome="errored")
    try:
        eval_meta = case_result.get("eval") or {}
        score = validate_score(eval_meta.get("avgScore"))
    except TypeError:
        return _observation(identity, **metrics, outcome="errored")
    if score is None:
        return _observation(identity, **metrics, outcome="unscored")
    return _observation(identity, **metrics, outcome="scored", score=score)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


class _Pair(TypedDict):
    control: Dict[str, Any]
    treatment: Dict[str, Any]


class _PairGroup:
    def __init__(self, eval_set: str, case_id: str, model: str, run_number: int) -> None:
        self.eval_set = eval_set
        self.case_id = case_id
        self.model = model
        self.run_number = run_number
        self.expected: Dict[str, int] = {}
        self.observations: Dict[str, List[EvalObservation]] = {}


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _difference(treatment: float, control: float) -> float:
    return float(f"{treatment - control:.15g}")


def _pair_key(identity: Mapping[str, Any]) -> tuple:
    return (identity["evalSet"], identity["caseId"], identity["model"], identity["runNumber"])


def _group_pairs(expected_runs: Sequence[Mapping[str, Any]],
                 observations: Sequence[EvalObservation]) -> List[_PairGroup]:
    groups: Dict[tuple, _PairGroup] = {}

    def get_group(identity: Mapping[str, Any]) -> _PairGroup:
        key = _pair_key(identity)
        existing = groups.get(key)
        if existing is None:
            existing = _PairGroup(identity["evalSet"], identity["caseId"],
                                  identity["model"], identity["runNumber"])
            groups[key] = existing
        return existing

    for expected in expected_runs:
        group = get_group(expected)
        variant = expected["variant"]
        group.expected[variant] = group.expected.get(variant, 0) + 1
    for observation in observations:
        group = get_group(observation)
        runs = group.observations.setdefault(observation["variant"], [])
        runs.append(observation)
    return sorted(groups.values(),
                  key=lambda group: (group.eval_set, group.case_id, group.model, group.run_number))


def _resolve_pair(group: _PairGroup) -> tuple:
    reasons: List[str] = []
    for variant in VARIANTS:
        expected = group.expected.get(variant, 0)
        observed = group.observations.get(variant, [])
        if expected != 1:
            reasons.append(f"{variant}: design expected 1 run, found {expected}")
        if len(observed) != expected:
            plural = "" if expected == 1 else "s"
            reasons.append(f"{variant}: expected {expected} observation{plural}, found {len(observed)}")
        if expected == 1 and len(observed) == 1 and observed[0]["outcome"] != "scored":
            reasons.append(f"{variant}: {observed[0]['outcome']}")
    if reasons:
        blocked: BlockedPair = {"evalSet": group.eval_set, "caseId": group.case_id,
                                "model": group.model, "runNumber": group.run_number, "reasons": reasons}
        return None, blocked
    pair: _Pair = {
        "control": group.observations[CONTROL][0],
        "treatment": group.observations[TREATMENT][0],
    }
    return pair, None


def _summarize_metric(pairs: Sequence[_Pair], select: Callable[[Mapping[str, Any]], Optional[float]]) -> PairedMetricSummary:
    control: List[float] = []
    treatment: List[float] = []
    for pair in pairs:
        control_value = select(pair["control"])
        treatment_value = select(pair["treatment"])
        if control_value is None or treatment_value is None:
            continue
        control.append(control_value)
        treatment.append(treatment_value)
    control_mean = _mean(control)
    treatment_mean = _mean(treatment)
    return {
        "eligiblePairs": len(control),
        "controlMean": control_mean,
        "treatmentMean": treatment_mean,
        "meanDelta": None if control_mean is None or treatment_mean is None else _difference(treatment_mean, control_mean),
    }


def _operational_total(runs: Sequence[Mapping[str, Any]],
                       select: Callable[[Mapping[str, Any]], Optional[float]]) -> OperationalMetricTotal:
    values = [value for run in runs if (value := select(run)) is not None]
    return {"availableRuns": len(values), "total": sum(values) if values else None}


def _variant_totals(observations: Sequence[EvalObservation], variant: str) -> VariantTotals:
    runs = [observation for observation in observations if observation["variant"] == variant]

    def field(name: str) -> Callable[[Mapping[str, Any]], Optional[float]]:
        return lambda observation: observation.get(name)

    return {
        "variant": variant,
        "runs": len(runs),
        "inputTokens": _operational_total(runs, field("inputTokens")),
        "outputTokens": _operational_total(runs, field("outputTokens")),
        "cacheReadTokens": _operational_total(runs, field("cacheReadTokens")),
        "cacheWriteTokens": _operational_total(runs, field("cacheWriteTokens")),
        "totalTokens": _operational_total(runs, field("totalTokens")),
        "toolCalls": _operational_total(runs, field("toolCalls")),
        "totalMs": _operational_total(runs, field("totalMs")),
        "estimatedCostUsd": _operational_total(runs, field("estimatedCostUsd")),
    }


def _comparison_flags(pairs: Sequence[_Pair], control_pass_rate: Optional[float],
                      treatment_pass_rate: Optional[float]) -> List[str]:
    flags: List[str] = []
    if control_pass_rate is not None and treatment_pass_rate is not None:
        if control_pass_rate == treatment_pass_rate:
            flags.append("no-lift")
        if treatment_pass_rate < control_pass_rate:
            flags.append("negative-delta")
        if control_pass_rate == 1:
            flags.append("control-saturated")
        if treatment_pass_rate == 1:
            flags.append("treatment-saturated")
    outcomes: Dict[tuple, set] = {}
    for pair in pairs:
        for side in ("control", "treatment"):
            observation = pair[side]
            key = (observation["caseId"], observation["variant"])
            outcomes.setdefault(key, set()).add(str(observation["score"] >= 1))
    if any(len(values) > 1 for values in outcomes.values()):
        flags.append("flaky")
    return flags


def summarize_eval_observations(
    protocol_digest: str,
    expected_runs: Sequence[Mapping[str, Any]],
    observations: Sequence[EvalObservation],
) -> EvalComparisonReport:
    """Pair exact arms and compute lift, flags, and operational totals."""
    groups = _group_pairs(expected_runs, observations)
    blocked_pairs: List[BlockedPair] = []
    pairs_by_eval_set: Dict[str, List[_Pair]] = {}
    totals_by_eval_set: Dict[str, int] = {}
    for group in groups:
        totals_by_eval_set[group.eval_set] = totals_by_eval_set.get(group.eval_set, 0) + 1
        pair, blocked = _resolve_pair(group)
        if blocked:
            blocked_pairs.append(blocked)
        if pair:
            pairs_by_eval_set.setdefault(group.eval_set, []).append(pair)
    comparisons: List[EvalSetComparison] = []
    for eval_set in sorted(totals_by_eval_set):
        total_pairs = totals_by_eval_set[eval_set]
        pairs = pairs_by_eval_set.get(eval_set, [])
        blocked_pair_count = total_pairs - len(pairs)
        publish_headline = blocked_pair_count == 0 and pairs
        control_pass_rate = (
            sum(1 for pair in pairs if pair["control"]["score"] >= 1) / len(pairs) if publish_headline else None
        )
        treatment_pass_rate = (
            sum(1 for pair in pairs if pair["treatment"]["score"] >= 1) / len(pairs) if publish_headline else None
        )
        lift = (None if control_pass_rate is None or treatment_pass_rate is None
                else _difference(treatment_pass_rate, control_pass_rate))
        comparisons.append({
            "evalSet": eval_set,
            "totalPairs": total_pairs,
            "eligiblePairs": len(pairs),
            "blockedPairs": blocked_pair_count,
            "controlPassRate": control_pass_rate,
            "treatmentPassRate": treatment_pass_rate,
            "lift": lift,
            "flags": _comparison_flags(pairs, control_pass_rate, treatment_pass_rate),
            "totalTokens": _summarize_metric(pairs, lambda observation: observation.get("totalTokens")),
            "toolCalls": _summarize_metric(pairs, lambda observation: observation.get("toolCalls")),
            "totalMs": _summarize_metric(pairs, lambda observation: observation.get("totalMs")),
            "estimatedCostUsd": _summarize_metric(pairs, lambda observation: observation.get("estimatedCostUsd")),
        })
    return {
        "schemaVersion": 3,
        "protocolDigest": protocol_digest,
        "control": CONTROL,
        "treatment": TREATMENT,
        "comparisons": comparisons,
        "blockedPairs": blocked_pairs,
        "operationalTotals": [_variant_totals(observations, CONTROL), _variant_totals(observations, TREATMENT)],
    }


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def _bold(text: str) -> str:
    if sys.stdout.isatty():
        return f"\x1b[1m{text}\x1b[22m"
    return text


def _percentage(value: Optional[float]) -> str:
    return "unavailable" if value is None else f"{value * 100:.1f}%"


def _signed(value: float, digits: int) -> str:
    return f"{'+' if value >= 0 else ''}{value:.{digits}f}"


def _paired_metric(label: str, metric: PairedMetricSummary, unit: str = "") -> str:
    if metric["meanDelta"] is None or metric["controlMean"] is None or metric["treatmentMean"] is None:
        return f"    {label.rjust(10)}  unavailable"
    return (f"    {label.rjust(10)}  {_signed(metric['meanDelta'], 1)}{unit}"
            f" (with {metric['treatmentMean']:.1f}{unit}, without {metric['controlMean']:.1f}{unit},"
            f" {metric['eligiblePairs']} pairs)")


def _operational_metric(metric: OperationalMetricTotal, runs: int, formatter: Callable[[float], str]) -> str:
    if metric["total"] is None:
        return f"unavailable (0/{runs} measured)"
    coverage = "" if metric["availableRuns"] == runs else f" ({metric['availableRuns']}/{runs} measured)"
    return f"{formatter(metric['total'])}{coverage}"


def format_eval_comparison_report(report: EvalComparisonReport) -> str:
    """Render the paired comparison report as text."""
    if not report["comparisons"]:
        return ""
    lines = [_bold("Documentation Eval Comparisons")]
    for comparison in report["comparisons"]:
        lines.append(f"  {comparison['evalSet']}")
        lines.append(f"         Pairs  {comparison['eligiblePairs']}/{comparison['totalPairs']} eligible")
        if comparison["lift"] is None:
            lines.append("     Pass rate  withheld because pairs are blocked"
                         if comparison["blockedPairs"] > 0 else "     Pass rate  unavailable")
        else:
            lines.append(
                f"     Pass rate  {_signed(comparison['lift'] * 100, 1)} pp"
                f" (with {_percentage(comparison['treatmentPassRate'])},"
                f" without {_percentage(comparison['controlPassRate'])})")
        if comparison["flags"]:
            lines.append(f"         Flags  {', '.join(comparison['flags'])}")
        lines.append(_paired_metric("Tokens", comparison["totalTokens"]))
        lines.append(_paired_metric("Tools", comparison["toolCalls"]))
        lines.append(_paired_metric("Latency", comparison["totalMs"], "ms"))
        cost = comparison["estimatedCostUsd"]
        if cost["meanDelta"] is None or cost["controlMean"] is None or cost["treatmentMean"] is None:
            lines.append("     Est. cost  unavailable")
        else:
            sign = "+" if cost["meanDelta"] >= 0 else "-"
            lines.append(
                f"     Est. cost  {sign}${abs(cost['meanDelta']):.4f}"
                f" (with ${cost['treatmentMean']:.4f}, without ${cost['controlMean']:.4f},"
                f" {cost['eligiblePairs']} pairs)")
    lines.append("  Operational totals")
    for totals in report["operationalTotals"]:
        tokens = _operational_metric(totals["totalTokens"], totals["runs"], lambda total: f"{total} tokens")
        tools = _operational_metric(totals["toolCalls"], totals["runs"], lambda total: f"{total} tools")
        latency = _operational_metric(totals["totalMs"], totals["runs"], lambda total: f"{total / 1000:.2f}s")
        cost = _operational_metric(totals["estimatedCostUsd"], totals["runs"], lambda total: f"${total:.4f} cost")
        lines.append(f"    {totals['variant']}: {totals['runs']} runs, {tokens}, {tools}, {latency}, {cost}")
    if report["blockedPairs"]:
        lines.append("  Blocked pairs")
        for blocked in report["blockedPairs"]:
            lines.append(
                f"    {blocked['evalSet']}/{blocked['caseId']}/{blocked['model']}/run-{blocked['runNumber']}:"
                f" {'; '.join(blocked['reasons'])}")
    return "\n".join(lines)
