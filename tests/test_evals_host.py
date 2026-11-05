"""Host eval runner: discovery, combined Vitest JSON reports, CLI behavior."""

import json

import pytest

from pi.evals.host import (
    discover_host_eval_files,
    format_report_summary,
    parse_arguments,
    run_host_eval_files,
)

SUITE = '''
from pi.evals.framework import create_harness, describe_eval

harness = create_harness("fixture", run=lambda args: {
    "output": {"ok": True},
    "events": [{"type": "message", "role": "user", "content": args.input},
               {"type": "tool_call", "id": "c1", "name": "echo", "arguments": {"x": 1}},
               {"type": "tool_result", "toolCallId": "c1", "name": "echo", "content": "done"}],
    "usage": {"provider": "fixture", "model": "model", "inputTokens": 10,
              "outputTokens": 5, "totalTokens": 15, "toolCalls": 1},
    "timings": {"totalMs": 50},
})


@describe_eval("Suite", harness=harness)
async def first_case(run):
    result = await run("produce the answer")
    assert result["errors"] == []


@describe_eval("Suite", harness=harness, title="second case")
async def second_case(run):
    calls = tool_calls(await run("call the tool"))
    assert [call["name"] for call in calls] == ["echo"]
'''


def write_suite(tmp_path, text=SUITE, name="suite.eval.py"):
    path = tmp_path / name
    path.write_text("from pi.evals.framework import describe_eval, tool_calls\n" + text, encoding="utf-8")
    return path


def test_discovery_lists_only_host_eval_files(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "a.eval.py").write_text("", encoding="utf-8")
    (tmp_path / "evals" / "b.docs.eval.py").write_text("", encoding="utf-8")
    (tmp_path / "evals" / "sub").mkdir()
    (tmp_path / "evals" / "sub" / "c.eval.py").write_text("", encoding="utf-8")
    (tmp_path / "evals" / "notes.md").write_text("", encoding="utf-8")

    files = discover_host_eval_files(tmp_path)
    names = [file.replace("\\", "/").split("/evals/")[-1] for file in files]
    assert names == ["a.eval.py", "sub/c.eval.py"]


def test_run_host_eval_files_runs_every_case_into_one_vitest_report(tmp_path):
    path = write_suite(tmp_path)
    report = _run(run_host_eval_files([str(path)]))
    assert report["numTotalTests"] == 2
    assert report["numPassedTests"] == 2
    assert report["numFailedTests"] == 0
    assert report["success"] is True
    assert len(report["testResults"]) == 1
    file_result = report["testResults"][0]
    assert file_result["name"] == str(path)
    assert file_result["status"] == "passed"
    assert [assertion["title"] for assertion in file_result["assertionResults"]] == [
        "first case", "second case"]


def test_run_host_eval_files_applies_the_name_filter(tmp_path):
    path = write_suite(tmp_path)
    report = _run(run_host_eval_files([str(path)], "second"))
    assert report["numTotalTests"] == 1
    assert report["testResults"][0]["assertionResults"][0]["fullName"] == "Suite > second case"


def test_run_host_eval_files_records_failures_without_aborting(tmp_path):
    path = write_suite(tmp_path, SUITE + '''

@describe_eval("Suite", harness=harness)
async def failing_case(run):
    result = await run("anything")
    assert result["output"] == {"ok": False}
''')
    report = _run(run_host_eval_files([str(path)]))
    assert report["numTotalTests"] == 3
    assert report["numFailedTests"] == 1
    assert report["success"] is False
    failed = report["testResults"][0]["assertionResults"][2]
    assert failed["status"] == "failed"
    assert failed["failureMessages"]


def test_parse_arguments_rejects_docs_files_and_unknown_flags():
    files, pattern, report = parse_arguments(["evals/smoke.eval.py", "-t", "Paris"])
    assert files == ["evals/smoke.eval.py"] and pattern == "Paris" and report is None
    files, pattern, report = parse_arguments(["evals/smoke.eval.py", "--report=out.json"])
    assert files == ["evals/smoke.eval.py"] and report == "out.json"
    with pytest.raises(ValueError, match="pi.evals.cli"):
        parse_arguments(["evals/examples.docs.eval.py"])
    with pytest.raises(ValueError, match="Unsupported host eval argument"):
        parse_arguments(["--wat"])
    with pytest.raises(ValueError, match="Missing value"):
        parse_arguments(["-t"])


def test_main_writes_a_vitest_json_report(tmp_path, monkeypatch, capsys):
    path = write_suite(tmp_path)
    monkeypatch.chdir(tmp_path)
    status = _main(["-t", "first", str(path), "--report", "out/vitest.json"])
    assert status == 0
    report = json.loads((tmp_path / "out" / "vitest.json").read_text(encoding="utf-8"))
    assert report["numTotalTests"] == 1 and report["success"] is True
    assert "1/1 passed" in capsys.readouterr().out


def test_main_fails_when_a_case_fails(tmp_path):
    path = write_suite(tmp_path, SUITE + '''

@describe_eval("Suite", harness=harness)
async def failing_case(run):
    result = await run("anything")
    assert result["output"] == {"ok": False}
''')
    assert _main([str(path)]) == 1
    assert "2/3 passed" in format_report_summary(_run(run_host_eval_files([str(path)])))


def test_main_reports_errors_as_exit_code_two():
    assert _main(["missing.eval.py"]) == 2


def _run(coroutine):
    import asyncio
    return asyncio.run(coroutine)


def _main(argv):
    import asyncio
    from pi.evals.host import main
    return main(list(argv))
