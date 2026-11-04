"""Documentation-lift eval runner CLI.

Port of ``packages/evals/src/cli.ts``. Runs ``python -m pi.evals.cli``: select
eval files, build the two variant images, discover cases in both containers,
plan the arms, run each arm isolated, and write the paired comparison report.

The TypeScript CLI runs Vitest inside the container; the Python CLI runs
``pi.evals.framework`` through ``docker/entrypoint.py`` and consumes the same
Vitest-schema JSON reports.
"""

from __future__ import annotations

import asyncio
import glob as glob_module
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .docker import build_images, create_docker_context, discover_cases, require_eval_auth_file, run_task
from .plan import DOCUMENTATION_VARIANTS, DiscoveredEvalCase, EvalTask, create_task_plan, parse_discovered_cases
from .report import (
    EvalObservation,
    errored_observation,
    format_eval_comparison_report,
    read_task_observation,
    summarize_eval_observations,
    task_identity,
)

package_root = Path(__file__).resolve().parents[2]

RUNNER_OPTIONS = {"--provider", "--model", "--runs-per-variant"}
FILTER_OPTIONS = {"-t", "--testNamePattern"}


@dataclass
class EvalCliOptions:
    provider: Optional[str] = None
    model: Optional[str] = None
    runs_per_variant: int = 1
    requested_files: List[str] = field(default_factory=list)
    discovery_args: List[str] = field(default_factory=list)


def parse_eval_cli(args: List[str], environment: Optional[Mapping[str, str]] = None) -> EvalCliOptions:
    """Parse runner and filter options; everything else is rejected."""
    environment = environment if environment is not None else os.environ
    provider: Optional[str] = None
    model: Optional[str] = None
    runs_text: Optional[str] = None
    cli_selected_model = False
    requested_files: List[str] = []
    discovery_args: List[str] = []

    index = 0
    while index < len(args):
        argument = args[index]
        if argument.endswith(".docs.eval.py"):
            requested_files.append(argument)
            index += 1
            continue
        if argument in FILTER_OPTIONS:
            if index + 1 >= len(args) or not args[index + 1]:
                raise ValueError(f"Missing value for {argument}.")
            discovery_args.extend([argument, args[index + 1]])
            index += 2
            continue
        if argument.startswith("--testNamePattern="):
            if len(argument) == len("--testNamePattern="):
                raise ValueError("Missing value for --testNamePattern.")
            discovery_args.append(argument)
            index += 1
            continue

        equals = argument.find("=")
        name = argument if equals == -1 else argument[:equals]
        if name in RUNNER_OPTIONS:
            if equals == -1:
                if index + 1 >= len(args) or not args[index + 1] or args[index + 1].startswith("-"):
                    raise ValueError(f"Missing value for {name}.")
                value = args[index + 1]
                index += 1
            else:
                value = argument[equals + 1:]
                if not value:
                    raise ValueError(f"Missing value for {name}.")
            if name == "--provider":
                provider = value
            elif name == "--model":
                model = value
            else:
                runs_text = value
            if name != "--runs-per-variant":
                cli_selected_model = True
            index += 1
            continue
        raise ValueError(f"Unsupported eval argument: {argument}")

    if provider is not None:
        provider = provider.strip() or None
    if model is not None:
        model = model.strip() or None
    if cli_selected_model:
        if not provider or not model:
            raise ValueError("CLI model selection requires both --provider and --model.")
    else:
        provider = (environment.get("PI_PROVIDER") or "").strip() or None
        model = (environment.get("PI_MODEL") or "").strip() or None
        if bool(provider) != bool(model):
            raise ValueError("Set both PI_PROVIDER and PI_MODEL, or neither.")

    configured_runs = (runs_text if runs_text is not None else environment.get("PI_EVAL_RUNS_PER_VARIANT"))
    configured_runs = configured_runs.strip() if configured_runs else ""
    try:
        runs_per_variant = int(configured_runs) if configured_runs else 1
    except ValueError:
        raise ValueError("Runs per variant must be a positive integer.") from None
    if runs_per_variant < 1:
        raise ValueError("Runs per variant must be a positive integer.")

    return EvalCliOptions(provider=provider, model=model, runs_per_variant=runs_per_variant,
                          requested_files=requested_files, discovery_args=discovery_args)


def artifact_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"
    return f"{stamp}_{uuid.uuid4()}"


def container_path(path: str) -> str:
    absolute = (package_root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    try:
        package_relative = absolute.relative_to(package_root).as_posix()
    except ValueError:
        raise ValueError(f"Eval file must be inside {package_root}: {path}") from None
    return package_relative


def normalize_discovered_file(path: str) -> str:
    prefix = "/repo/pi-python/"
    if not path.startswith(prefix):
        raise ValueError(f"Discovered eval path is outside the container package: {path}")
    return path[len(prefix):]


def compare_discovery(left: List[DiscoveredEvalCase], right: List[DiscoveredEvalCase]) -> None:
    def identities(cases: List[DiscoveredEvalCase]) -> str:
        return json.dumps(sorted(json.dumps([case.full_name, normalize_discovered_file(case.file)],
                                            ensure_ascii=False) for case in cases), ensure_ascii=False)

    if identities(left) != identities(right):
        raise ValueError("Documentation variants discovered different eval cases.")


async def run(argv: List[str]) -> int:
    cli = parse_eval_cli(argv)
    if not cli.provider or not cli.model:
        raise ValueError("Set PI_PROVIDER and PI_MODEL, or pass --provider and --model.")
    selected_model = {"provider": cli.provider, "model": cli.model}
    model_identity = f"{selected_model['provider']}/{selected_model['model']}"
    if cli.requested_files:
        selected = cli.requested_files
    else:
        selected = [os.path.relpath(match, package_root).replace(os.sep, "/")
                    for match in glob_module.glob(str(package_root / "evals" / "**" / "*.docs.eval.py"), recursive=True)]
    files = sorted(container_path(path) for path in selected)
    if not files:
        raise ValueError("No documentation eval files were selected.")
    for file in files:
        if not file.endswith(".docs.eval.py"):
            raise ValueError(f"Documentation runner cannot execute non-doc eval: {file}")

    artifact_directory = (package_root / ".eval" / artifact_run_id()).resolve()
    auth_path = require_eval_auth_file(selected_model["provider"])
    os.makedirs(artifact_directory, mode=0o700, exist_ok=True)
    os.chmod(artifact_directory, 0o700)
    images = build_images()
    if images["with_docs"]["id"] == images["without_docs"]["id"]:
        raise ValueError("Documentation variants resolved to the same image.")
    docker = create_docker_context(images, str(artifact_directory), auth_path,
                                   selected_model["provider"], selected_model["model"], cli.runs_per_variant)

    discoveries: List[List[DiscoveredEvalCase]] = []
    for variant in DOCUMENTATION_VARIANTS:
        path = discover_cases(docker, variant, files, cli.discovery_args)
        with open(path, "r", encoding="utf-8") as handle:
            discoveries.append(parse_discovered_cases(json.load(handle)))
    compare_discovery(discoveries[0], discoveries[1])
    cases = [DiscoveredEvalCase(file=normalize_discovered_file(case.file), full_name=case.full_name,
                                eval_set=case.eval_set, case_id=case.case_id)
             for case in discoveries[0]]
    if not cases:
        raise ValueError("No documentation eval cases matched the selection.")
    tasks = create_task_plan(cases, model_identity, cli.runs_per_variant)

    def task_json(task: EvalTask) -> Dict[str, Any]:
        return {"file": task.file, "fullName": task.full_name, "evalSet": task.eval_set,
                "caseId": task.case_id, "variant": task.variant, "model": task.model,
                "runNumber": task.run_number}

    protocol = {
        "schemaVersion": 1,
        "model": model_identity,
        "runsPerVariant": cli.runs_per_variant,
        "images": images,
        "files": files,
        "cases": [{"evalSet": case.eval_set, "caseId": case.case_id, "file": case.file} for case in cases],
        "tasks": [task_json(task) for task in tasks],
    }
    protocol_text = json.dumps(protocol, ensure_ascii=False, separators=(",", ":"))
    protocol_digest = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    (artifact_directory / "protocol.json").write_text(
        json.dumps({**protocol, "protocolDigest": protocol_digest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (artifact_directory / "expected-runs.json").write_text(
        json.dumps([task_json(task) for task in tasks], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    observations: List[EvalObservation] = []
    for index, task in enumerate(tasks):
        print(f"\n[{index + 1}/{len(tasks)}] {task.eval_set} > {task.case_id} | {task.variant}"
              f" | {task.model} | run {task.run_number}", flush=True)
        report_path = run_task(docker, task)
        observation = (await read_task_observation(task, report_path, str(artifact_directory))
                       if report_path else errored_observation(task))
        observations.append(observation)
        (artifact_directory / "observations.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in observations),
            encoding="utf-8")

    report = summarize_eval_observations(protocol_digest, [task_identity(task) for task in tasks], observations)
    report_text = format_eval_comparison_report(report)
    (artifact_directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    (artifact_directory / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    print(f"\n{report_text}")
    print(f"\nArtifacts: {artifact_directory}")
    return 1 if report["blockedPairs"] else 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return asyncio.run(run(list(sys.argv[1:] if argv is None else argv)))
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
