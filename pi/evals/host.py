"""Host eval runner.

The TypeScript host evals run on the developer machine through
``vitest run --project host`` (``npm run eval:host``); this module is the
Python replacement: it discovers ``evals/**/*.eval.py`` files that are not
``*.docs.eval.py``, runs every selected case sequentially through
``pi.evals.framework``, and writes the same Vitest-schema JSON report.

Usage::

    python -m pi.evals.host [file.eval.py ...] [-t PATTERN] [--report PATH]

Without file arguments every host eval file under ``evals/`` runs. The model
comes from PI_PROVIDER / PI_MODEL, like the documentation runner.
"""

from __future__ import annotations

import asyncio
import glob as glob_module
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .framework import load_cases_from_file, run_eval_case

package_root = Path(__file__).resolve().parents[2]


def discover_host_eval_files(root: Optional[Path] = None) -> List[str]:
    """Every host eval file (``*.eval.py`` excluding ``*.docs.eval.py``), sorted."""
    root = Path(root) if root is not None else package_root
    matches = glob_module.glob(str(root / "evals" / "**" / "*.eval.py"), recursive=True)
    return sorted(match for match in matches if not match.endswith(".docs.eval.py"))


def select_cases(cases: List[Any], test_name_pattern: Optional[str]) -> List[Any]:
    if test_name_pattern is None:
        return list(cases)
    return [case for case in cases if re.search(test_name_pattern, case.full_name)]


async def run_host_eval_files(files: List[str], test_name_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Run every selected case and assemble the combined Vitest JSON report."""
    started_ms = int(time.time() * 1000)
    test_results: List[Dict[str, Any]] = []
    num_passed = 0
    num_failed = 0
    for file in files:
        cases = select_cases(load_cases_from_file(file), test_name_pattern)
        assertions = [await run_eval_case(case) for case in cases]
        passed = sum(1 for assertion in assertions if assertion["status"] == "passed")
        failed = len(assertions) - passed
        num_passed += passed
        num_failed += failed
        test_results.append({
            "name": file,
            "message": "",
            "status": "failed" if failed else "passed",
            "assertionResults": assertions,
        })
    return {
        "numFailedTests": num_failed,
        "numPassedTests": num_passed,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "numTotalTests": num_passed + num_failed,
        "startTime": started_ms,
        "success": num_failed == 0,
        "testResults": test_results,
    }


def parse_arguments(argv: List[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Split eval file paths, the case filter, and the report path."""
    files: List[str] = []
    test_name_pattern: Optional[str] = None
    report_path: Optional[str] = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("-t", "--testNamePattern"):
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise ValueError(f"Missing value for {argument}.")
            test_name_pattern = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--testNamePattern="):
            value = argument[len("--testNamePattern="):]
            if not value:
                raise ValueError("Missing value for --testNamePattern.")
            test_name_pattern = value
            index += 1
            continue
        if argument == "--report":
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise ValueError("Missing value for --report.")
            report_path = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--report="):
            value = argument[len("--report="):]
            if not value:
                raise ValueError("Missing value for --report.")
            report_path = value
            index += 1
            continue
        if argument.endswith(".eval.py"):
            if argument.endswith(".docs.eval.py"):
                raise ValueError(
                    f"Documentation eval files run through python -m pi.evals.cli: {argument}")
            files.append(argument)
            index += 1
            continue
        raise ValueError(f"Unsupported host eval argument: {argument}")
    return files, test_name_pattern, report_path


def resolve_eval_path(path: str) -> str:
    """Resolve an eval file path relative to the package root."""
    absolute = Path(path) if os.path.isabs(path) else package_root / path
    return str(absolute)


def format_report_summary(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    for result in report["testResults"]:
        for assertion in result["assertionResults"]:
            lines.append(f"{'PASS' if assertion['status'] == 'passed' else 'FAIL'}  {assertion['fullName']}")
            for failure in assertion["failureMessages"]:
                lines.append(f"      - {failure}")
    lines.append(f"{report['numPassedTests']}/{report['numTotalTests']} passed")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        files, test_name_pattern, report_path = parse_arguments(argv)
        if not files:
            files = discover_host_eval_files()
        paths = [resolve_eval_path(path) for path in files]
        report = asyncio.run(run_host_eval_files(paths, test_name_pattern))
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, SyntaxError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(format_report_summary(report), flush=True)
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {destination}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
