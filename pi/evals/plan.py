"""Task planning for documentation-lift evals.

Port of ``packages/evals/src/plan.ts``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Sequence

DOCUMENTATION_VARIANTS = ("without_docs", "with_docs")
DocumentationVariant = str  # "without_docs" | "with_docs"


@dataclass
class DiscoveredEvalCase:
    file: str
    full_name: str
    eval_set: str
    case_id: str


@dataclass
class EvalTask(DiscoveredEvalCase):
    variant: str
    model: str
    run_number: int


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def parse_discovered_cases(value: Any) -> List[DiscoveredEvalCase]:
    """Parse ``{"name", "file"}`` discovery records into stable case identities."""
    if not isinstance(value, list):
        raise TypeError("Discovered eval cases must be an array.")
    identities = set()
    cases: List[DiscoveredEvalCase] = []
    for item in value:
        if not _is_record(item) or not isinstance(item.get("name"), str) or not isinstance(item.get("file"), str):
            raise TypeError("Discovered eval case is invalid.")
        parts = item["name"].split(" > ")
        eval_set, case_id, extra = (parts + ["", ""])[:3]
        if not eval_set.strip() or not case_id.strip() or len(parts) > 2:
            raise TypeError(f'Documentation eval must use "<eval set> > <case>": {item["name"]}')
        identity = json.dumps([eval_set, case_id], ensure_ascii=False)
        if identity in identities:
            raise TypeError(f"Duplicate eval case identity: {item['name']}")
        identities.add(identity)
        cases.append(DiscoveredEvalCase(file=item["file"], full_name=item["name"],
                                        eval_set=eval_set, case_id=case_id))
    return cases


def create_task_plan(cases: Sequence[DiscoveredEvalCase], model: str, runs_per_variant: int) -> List[EvalTask]:
    """Expand cases into ``(case, variant, model, repetition)`` tasks."""
    if not model or "/" not in model or model.startswith("/") or model.endswith("/"):
        raise TypeError("Model identity must contain a provider and model.")
    if not isinstance(runs_per_variant, int) or isinstance(runs_per_variant, bool) or runs_per_variant < 1:
        raise TypeError("Runs per variant must be a positive integer.")
    tasks: List[EvalTask] = []
    for eval_case in cases:
        for run_number in range(1, runs_per_variant + 1):
            variants: List[DocumentationVariant] = (
                ["without_docs", "with_docs"] if run_number % 2 == 1 else ["with_docs", "without_docs"]
            )
            for variant in variants:
                tasks.append(EvalTask(file=eval_case.file, full_name=eval_case.full_name,
                                      eval_set=eval_case.eval_set, case_id=eval_case.case_id,
                                      variant=variant, model=model, run_number=run_number))
    return tasks
