"""Evaluation runner.

Defines eval cases (prompt + checks), runs them against any async runner
(usually a live agent or a faux provider), and scores the transcript.
Checkers: contains, not_contains, regex, response_equals, tool_called,
no_tool_called, max_tokens, max_tool_calls. Custom checks are callables
receiving the result dict and returning an error string or None.

This package also carries the documentation-lift eval system ported from
``packages/evals``: ``pi.evals.harness`` (isolated agent harness),
``pi.evals.plan`` (task planning), ``pi.evals.report`` (paired comparison),
``pi.evals.docker`` (container orchestration), ``pi.evals.cli`` (runner CLI),
and ``pi.evals.framework`` (the local vitest-evals replacement).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ..ai.types import AssistantMessage, TextContent, ToolCall

Check = Union[dict, Callable[[dict], Optional[str]]]

CHECK_TYPES = ("contains", "not_contains", "regex", "response_equals", "tool_called",
               "no_tool_called", "max_tokens", "max_tool_calls")


@dataclass
class EvalCase:
    name: str
    prompt: str
    checks: List[Check] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> "EvalCase":
        if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not isinstance(data.get("prompt"), str):
            raise ValueError("Eval case requires name and prompt strings")
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            raise ValueError("Eval checks must be a list")
        for check in checks:
            if isinstance(check, dict):
                if check.get("type") not in CHECK_TYPES:
                    raise ValueError(f"Unknown eval check type: {check.get('type')!r}")
            elif not callable(check):
                raise ValueError("Eval checks must be objects or callables")
        return cls(name=data["name"], prompt=data["prompt"], checks=checks)


@dataclass
class EvalResult:
    name: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    tokens: int = 0
    tool_calls: int = 0
    duration: float = 0.0


def _flatten(messages: list) -> dict:
    texts: List[str] = []
    tool_names: List[str] = []
    tokens = 0
    tool_call_count = 0
    for message in messages:
        usage = getattr(message, "usage", None)
        if usage is not None:
            tokens += usage.total_tokens
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextContent):
                    texts.append(block.text)
                elif isinstance(block, ToolCall):
                    tool_names.append(block.name)
                    tool_call_count += 1
        else:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(block.text for block in content if isinstance(block, TextContent))
    return {"text": "\n".join(texts), "tool_names": tool_names,
            "tokens": tokens, "tool_calls": tool_call_count}


def check_case(case: EvalCase, transcript: dict) -> List[str]:
    failures: List[str] = []
    text, tool_names = transcript["text"], transcript["tool_names"]
    for check in case.checks:
        if callable(check):
            error = check(transcript)
            if error:
                failures.append(str(error))
            continue
        kind = check.get("type")
        if kind == "contains":
            if check.get("text", "") not in text:
                failures.append(f"expected output to contain: {check.get('text')!r}")
        elif kind == "not_contains":
            if check.get("text", "") in text:
                failures.append(f"expected output not to contain: {check.get('text')!r}")
        elif kind == "regex":
            if not re.search(check.get("pattern", ""), text):
                failures.append(f"expected output to match: {check.get('pattern')!r}")
        elif kind == "response_equals":
            if text.strip() != str(check.get("text", "")).strip():
                failures.append(f"expected exact response: {check.get('text')!r}")
        elif kind == "tool_called":
            name = check.get("name")
            if name and name not in tool_names:
                failures.append(f"expected tool call: {name}")
            elif not tool_names:
                failures.append("expected at least one tool call")
        elif kind == "no_tool_called":
            if tool_names:
                failures.append(f"expected no tool calls, got: {', '.join(tool_names)}")
        elif kind == "max_tokens":
            if transcript["tokens"] > check.get("limit", 0):
                failures.append(f"expected <= {check.get('limit')} tokens, got {transcript['tokens']}")
        elif kind == "max_tool_calls":
            if transcript["tool_calls"] > check.get("limit", 0):
                failures.append(f"expected <= {check.get('limit')} tool calls, got {transcript['tool_calls']}")
    return failures


async def run_case(case: EvalCase, runner: Callable) -> EvalResult:
    """Run one case; ``runner(prompt)`` returns the transcript message list."""
    started = time.monotonic()
    try:
        outcome = runner(case.prompt)
        messages = await outcome if asyncio.iscoroutine(outcome) else await asyncio.wait_for(outcome, 600)
        transcript = _flatten(list(messages))
        failures = check_case(case, transcript)
        return EvalResult(name=case.name, passed=not failures, failures=failures,
                          tokens=transcript["tokens"], tool_calls=transcript["tool_calls"],
                          duration=time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - eval harness records failures
        return EvalResult(name=case.name, passed=False, failures=[f"{type(exc).__name__}: {exc}"],
                          duration=time.monotonic() - started)


async def run_evals(cases: List[EvalCase], runner: Callable, concurrency: int = 1) -> List[EvalResult]:
    """Run cases with bounded concurrency, preserving input order."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(case: EvalCase) -> EvalResult:
        async with semaphore:
            return await run_case(case, runner)

    return list(await asyncio.gather(*(guarded(case) for case in cases)))


def load_cases(path: Union[str, Path]) -> List[EvalCase]:
    """Load cases from a JSON file: a list or {"cases": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("Eval file must contain a list of cases")
    return [EvalCase.from_json(entry) for entry in data]


def summarize(results: List[EvalResult]) -> str:
    lines = []
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        lines.append(f"{marker}  {result.name}  ({result.tokens} tokens, {result.duration:.1f}s)")
        for failure in result.failures:
            lines.append(f"      - {failure}")
    passed = sum(1 for result in results if result.passed)
    lines.append(f"{passed}/{len(results)} passed")
    return "\n".join(lines)


# -- Documentation-lift eval system (port of packages/evals) ----------------

from .framework import (  # noqa: E402
    EvalCaseDef,
    Harness,
    HarnessContext,
    HarnessRunArgs,
    HarnessRunError,
    StructuredOutputJudge,
    attach_harness_run_to_error,
    create_harness,
    describe_eval,
    get_harness_run_from_error,
    normalize_harness_run,
    structured_output_judge,
    to_json_value,
)
from .harness import (  # noqa: E402
    DOCUMENTATION_EVAL_TOOLS,
    PiCodingAgentHarnessOptions,
    ModelSelection,
    apply_isolated_environment,
    create_pi_coding_agent_harness,
    create_pi_documentation_eval_harness,
    exclude_pi_documentation,
    resolve_documentation_variant,
    resolve_model_selection,
    verify_system_prompt,
)
from .plan import (  # noqa: E402
    DOCUMENTATION_VARIANTS,
    DiscoveredEvalCase,
    EvalTask,
    create_task_plan,
    parse_discovered_cases,
)
from .report import (  # noqa: E402
    EvalComparisonReport,
    EvalObservation,
    PI_SESSION_SNAPSHOT_ARTIFACT,
    classify_case_status,
    errored_observation,
    format_eval_comparison_report,
    read_task_observation,
    summarize_eval_observations,
)
