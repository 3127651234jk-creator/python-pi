"""Framework essentials: judge, describe_eval, run normalization, discovery."""

import asyncio
import json

import pytest

from pi.evals.framework import (
    HarnessRunError,
    StructuredOutputJudge,
    attach_harness_run_to_error,
    create_harness,
    discover_eval_files,
    normalize_harness_run,
    run_eval_file,
    structured_output_judge,
    to_json_value,
    tool_calls,
)

SUITE_TEMPLATE = '''
from pi.evals.framework import create_harness, describe_eval, structured_output_judge

harness = create_harness("fixture", run=lambda args: {
    "output": {"ok": True},
    "events": [{"type": "message", "role": "user", "content": args.input}],
    "usage": {"provider": "fixture", "model": "model", "inputTokens": 10,
              "outputTokens": 5, "totalTokens": 15, "toolCalls": 0},
    "timings": {"totalMs": 50},
})

judge = structured_output_judge(expected={"ok": True}, match="strict", allow_extras=False)


@describe_eval("Suite", harness=harness, judges=[judge], judge_threshold=JUDGE_THRESHOLD)
async def passes_the_check(run):
    await run("produce the answer")
'''


def write_suite(tmp_path, judge_threshold="None"):
    path = tmp_path / "suite.docs.eval.py"
    path.write_text(SUITE_TEMPLATE.replace("JUDGE_THRESHOLD", judge_threshold), encoding="utf-8")
    return path


def test_to_json_value_drops_non_json_values():
    assert to_json_value({"a": 1, "b": object(), "c": [1, float("nan"), "x"], "d": None}) == {
        "a": 1, "c": [1, None, "x"], "d": None,
    }


def test_normalize_harness_run_fills_defaults():
    run = normalize_harness_run("input", {"output": {"ok": True}, "events": []})
    assert run["input"] == "input"
    assert run["output"] == {"ok": True}
    assert run["session"]["events"] == []
    assert run["usage"] == {}
    assert run["timings"] == {}
    assert run["artifacts"] == {}
    assert run["errors"] == []


def test_normalize_harness_run_merges_context_artifacts():
    from pi.evals.framework import HarnessContext

    context = HarnessContext()
    context.set_artifact("runId", "run-1")
    run = normalize_harness_run("in", {"events": [], "artifacts": {"a": 1}}, context)
    assert run["artifacts"] == {"a": 1, "runId": "run-1"}


def test_attach_and_recover_harness_run():
    run = {"session": {"events": []}}
    error = attach_harness_run_to_error(RuntimeError("boom"), run)
    assert isinstance(error, HarnessRunError)
    assert error.harness_run is run
    assert "boom" in str(error)


def test_structured_output_judge_strict_match():
    judge = structured_output_judge(expected={"ok": True}, match="strict", allow_extras=False)
    assert judge.assess(output='{"ok": true}')["score"] == 1
    assert judge.assess(output='{"ok": 1}')["score"] == 0
    assert judge.assess(output='{"ok": true, "extra": 1}')["score"] == 0
    assert judge.assess(output="not json")["score"] == 0
    assert judge.assess(output='{"error": "boom"}')["score"] == 0
    assert judge.assess(output='{"ok": true}', expected={})["score"] == 1


def test_structured_output_judge_rationales():
    judge = structured_output_judge(expected={"ok": True}, match="strict", allow_extras=False)
    outcome = judge.assess(output='{"ok": 1}')
    assert "ok" in outcome["metadata"]["rationale"]
    outcome = judge.assess(output='{"ok": true, "more": 2}')
    assert "extra fields" in outcome["metadata"]["rationale"]


def test_run_eval_file_scores_the_case(tmp_path):
    path = write_suite(tmp_path)
    report = asyncio.run(run_eval_file(str(path)))
    assert report["numTotalTests"] == 1
    assert report["success"] is True
    assertion = report["testResults"][0]["assertionResults"][0]
    assert assertion["fullName"] == "Suite > passes the check"
    assert assertion["status"] == "passed"
    assert assertion["meta"]["eval"]["avgScore"] == 1
    assert assertion["meta"]["eval"]["thresholdFailed"] is False
    harness_run = assertion["meta"]["harness"]["run"]
    assert harness_run["output"] == {"ok": True}
    assert harness_run["usage"]["provider"] == "fixture"
    assert harness_run["session"]["events"][0]["content"] == "produce the answer"


def test_run_eval_file_applies_the_judge_threshold(tmp_path):
    path = write_suite(tmp_path, judge_threshold="2")
    report = asyncio.run(run_eval_file(str(path)))
    assert report["success"] is False
    assertion = report["testResults"][0]["assertionResults"][0]
    assert assertion["status"] == "failed"
    assert assertion["meta"]["eval"]["thresholdFailed"] is True
    assert assertion["failureMessages"]


def test_discover_eval_files_lists_case_identities(tmp_path):
    path = write_suite(tmp_path)
    records = discover_eval_files([str(path)])
    assert records == [{"name": "Suite > passes the check", "file": str(path)}]


def test_run_eval_file_requires_exactly_one_matching_case(tmp_path):
    path = write_suite(tmp_path)
    with pytest.raises(ValueError, match="exactly one case"):
        asyncio.run(run_eval_file(str(path), "no-such-case"))


TITLE_SUITE = '''
from pi.evals.framework import create_harness, describe_eval

harness = create_harness("fixture", run=lambda args: {"output": "ok", "events": []})


@describe_eval("Suite", harness=harness, title="page.md matches the implementation")
async def any_function_name(run):
    await run("x")
'''


def test_describe_eval_honors_an_explicit_case_title(tmp_path):
    path = tmp_path / "suite.eval.py"
    path.write_text(TITLE_SUITE, encoding="utf-8")
    records = discover_eval_files([str(path)])
    assert records == [{"name": "Suite > page.md matches the implementation", "file": str(path)}]


def test_tool_calls_pair_calls_with_results():
    run = {"session": {"events": [
        {"type": "tool_call", "id": "c1", "name": "read", "arguments": {"path": "a"}},
        {"type": "tool_call", "id": "c2", "name": "grep", "arguments": {}},
        {"type": "tool_call", "id": "c3", "name": "ls"},
        {"type": "tool_result", "toolCallId": "c1", "name": "read", "content": "text"},
        {"type": "tool_result", "toolCallId": "c2", "name": "grep", "content": "",
         "error": {"message": "boom"}},
    ]}}
    assert tool_calls(run) == [
        {"name": "read", "arguments": {"path": "a"}, "status": "ok", "result": "text"},
        {"name": "grep", "arguments": {}, "status": "error", "error": {"message": "boom"}},
        {"name": "ls", "status": "pending"},
    ]


def test_tool_calls_accepts_a_raw_event_list_or_session():
    events = [{"type": "tool_call", "id": "c1", "name": "read"}]
    assert tool_calls(events) == tool_calls({"session": {"events": events}}) == [
        {"name": "read", "status": "pending"}]
