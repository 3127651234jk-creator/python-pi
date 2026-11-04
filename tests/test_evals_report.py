"""Port of packages/evals/test/report.test.ts."""

import asyncio
import json

import pytest

from pi.evals.plan import EvalTask
from pi.evals.report import (
    PI_SESSION_SNAPSHOT_ARTIFACT,
    classify_case_status,
    read_task_observation,
)

TASK = EvalTask(
    file="evals/example.docs.eval.py",
    full_name="Example workflow > handles the case",
    eval_set="Example workflow",
    case_id="handles the case",
    variant="without_docs",
    model="fixture/model",
    run_number=1,
)
SESSION = '{"type":"session"}\n'


def write_task_report(directory, assertion=None):
    assertion = assertion or {}
    status = assertion.get("status", "passed")
    report = {
        "numFailedTests": 0,
        "numPassedTests": 1 if status == "passed" else 0,
        "numPendingTests": 1 if status in ("pending", "skipped") else 0,
        "numTodoTests": 0,
        "numTotalTests": 1,
        "startTime": 0,
        "success": True,
        "testResults": [{
            "message": "",
            "name": "/repo/pi-python/evals/example.docs.eval.py",
            "status": "passed",
            "assertionResults": [{
                "ancestorTitles": [TASK.eval_set],
                "fullName": f"{TASK.eval_set} {TASK.case_id}",
                "status": status,
                "title": TASK.case_id,
                "failureMessages": [],
                "meta": assertion.get("meta", {}),
            }],
        }],
    }
    path = directory / "vitest.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def scored_meta(overrides=None):
    overrides = overrides or {}
    return {
        "eval": {
            "avgScore": overrides.get("avgScore", 0.5) if "avgScore" in overrides else 0.5,
            "scores": [{"name": "StructuredOutputJudge", "score": 0.5}],
            "thresholdFailed": False,
        },
        "harness": {
            "name": "without_docs",
            "run": {
                "output": {"ok": True},
                "session": {"events": [{"type": "message", "role": "user", "content": "prompt"}]},
                "usage": {
                    "provider": "fixture",
                    "model": overrides.get("model", "model"),
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15,
                    "toolCalls": 1,
                    "metadata": {"cacheReadTokens": 2, "cacheWriteTokens": 3, "estimatedCostUsd": 0.01},
                },
                "timings": {"totalMs": 1234},
                "artifacts": overrides.get("artifacts", {"runId": "run-1", PI_SESSION_SNAPSHOT_ARTIFACT: SESSION}),
                "errors": overrides.get("errors", []),
            },
        },
    }


def read_observation(directory, assertion=None):
    report_path = write_task_report(directory, assertion)
    return asyncio.run(read_task_observation(TASK, str(report_path), str(directory)))


def identity_fields():
    return {"evalSet": TASK.eval_set, "caseId": TASK.case_id, "variant": TASK.variant,
            "model": TASK.model, "runNumber": TASK.run_number}


@pytest.mark.parametrize("status", ["skipped", "todo", "disabled"])
def test_maps_status_to_skipped(tmp_path, status):
    assert classify_case_status(status) == "skipped"


def test_maps_failed_infrastructure_to_errored(tmp_path):
    assert classify_case_status("failed") == "errored"


def test_preserves_a_skipped_outcome_when_no_harness_run_exists(tmp_path):
    observation = read_observation(tmp_path, {"status": "skipped"})
    assert observation["outcome"] == "skipped"


def test_preserves_a_pending_outcome_when_no_harness_run_exists(tmp_path):
    observation = read_observation(tmp_path, {"status": "pending"})
    assert observation["outcome"] == "pending"


def test_preserves_metrics_from_a_failed_eval_with_a_partial_harness_run(tmp_path):
    observation = read_observation(tmp_path, {
        "status": "failed",
        "meta": scored_meta({"errors": [{"message": "Prompt verification failed"}]}),
    })
    assert observation == {
        **identity_fields(),
        "outcome": "errored",
        "inputTokens": 10,
        "outputTokens": 5,
        "cacheReadTokens": 2,
        "cacheWriteTokens": 3,
        "totalTokens": 15,
        "toolCalls": 1,
        "totalMs": 1234,
        "estimatedCostUsd": 0.01,
    }
    sessions = (tmp_path / TASK.variant / "sessions").iterdir()
    session_file = next(iter(sessions)) / "session.jsonl"
    assert session_file.read_text(encoding="utf-8") == SESSION


def test_records_an_errored_outcome_when_a_passed_eval_has_no_harness_run(tmp_path):
    observation = read_observation(tmp_path, {"status": "passed"})
    assert observation["outcome"] == "errored"


def test_reads_a_scored_harness_run_and_persists_the_session_artifact(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta()})
    assert observation == {
        **identity_fields(),
        "outcome": "scored",
        "score": 0.5,
        "inputTokens": 10,
        "outputTokens": 5,
        "cacheReadTokens": 2,
        "cacheWriteTokens": 3,
        "totalTokens": 15,
        "toolCalls": 1,
        "totalMs": 1234,
        "estimatedCostUsd": 0.01,
    }
    hashes = list((tmp_path / TASK.variant / "sessions").iterdir())
    assert len(hashes) == 1
    assert (hashes[0] / "session.jsonl").read_text(encoding="utf-8") == SESSION


def test_treats_a_zero_score_as_scored_data(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta({"avgScore": 0})})
    assert observation["outcome"] == "scored"
    assert observation["score"] == 0


def test_records_an_unscored_outcome_when_a_completed_eval_has_no_score(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta({"avgScore": None})})
    assert observation["outcome"] == "unscored"


def test_records_an_errored_outcome_when_the_reported_model_does_not_match(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta({"model": "other"})})
    assert observation["outcome"] == "errored"


def test_records_an_errored_outcome_when_a_completed_harness_run_contains_errors(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta({"errors": [{"message": "boom"}]})})
    assert observation["outcome"] == "errored"


def test_still_scores_a_completed_eval_when_the_session_artifact_is_missing(tmp_path):
    observation = read_observation(tmp_path, {"meta": scored_meta({"artifacts": {"runId": "run-1"}})})
    assert observation["outcome"] == "scored"
    assert observation["score"] == 0.5
