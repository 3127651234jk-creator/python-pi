"""Port of packages/evals/test/plan.test.ts."""

import pytest

from pi.evals.plan import create_task_plan, parse_discovered_cases

DISCOVERED = [{"name": "Add model > adds the model", "file": "evals/models.docs.eval.py"}]


def test_derives_stable_case_identity_from_ordinary_names():
    cases = parse_discovered_cases(DISCOVERED)
    assert [(case.full_name, case.eval_set, case.case_id, case.file) for case in cases] == [
        ("Add model > adds the model", "Add model", "adds the model", "evals/models.docs.eval.py"),
    ]


def test_rejects_ambiguous_and_duplicate_identities():
    with pytest.raises(TypeError, match="<eval set> > <case>"):
        parse_discovered_cases([{"name": "adds the model", "file": "model.py"}])
    with pytest.raises(TypeError, match="Duplicate eval case identity"):
        parse_discovered_cases(DISCOVERED + DISCOVERED)


def test_creates_one_isolated_task_per_case_variant_model_and_repetition():
    cases = parse_discovered_cases(DISCOVERED)
    tasks = create_task_plan(cases, "fixture/model", 2)
    assert len(tasks) == 4
    assert [(task.variant, task.run_number) for task in tasks] == [
        ("without_docs", 1),
        ("with_docs", 1),
        ("with_docs", 2),
        ("without_docs", 2),
    ]


def test_rejects_invalid_model_identities_and_repetitions():
    cases = parse_discovered_cases(DISCOVERED)
    with pytest.raises(TypeError, match="provider and model"):
        create_task_plan(cases, "model", 1)
    with pytest.raises(TypeError, match="positive integer"):
        create_task_plan(cases, "fixture/model", 0)
