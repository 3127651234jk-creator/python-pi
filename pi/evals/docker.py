"""Docker orchestration for documentation-lift evals.

Port of ``packages/evals/src/docker.ts``. Builds the two variant images from
``docker/Dockerfile`` next to this package, requires the host auth.json, and
runs discovery and single-task containers. The TypeScript Vite tmpfs mount has
no Python equivalent and is omitted (the Python image writes no bytecode).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .plan import DocumentationVariant, EvalTask

package_root = Path(__file__).resolve().parents[2]
repository_root = package_root.parent

BuiltImages = Dict[str, Dict[str, str]]  # variant -> {name, id}


@dataclass
class DockerContext:
    images: BuiltImages
    artifact_directory: str
    auth_path: str
    provider: str
    model: str
    runs_per_variant: int


def _execute(command: str, args: List[str], capture: bool = False) -> Tuple[int, str]:
    result = subprocess.run(
        [command, *args],
        cwd=str(package_root),
        stdout=subprocess.PIPE if capture else None,
        text=True,
        encoding="utf-8",
    )
    return result.returncode, (result.stdout or "") if capture else ""


def _require_success(command: str, args: List[str]) -> None:
    status, _ = _execute(command, args)
    if status != 0:
        raise RuntimeError(f"{command} exited with status {status}.")


def build_images() -> BuiltImages:
    """Build and inspect the without_docs and with_docs images."""
    prefix = f"pi-evals-{hashlib.sha256(str(repository_root).encode('utf-8')).hexdigest()[:12]}"
    images: BuiltImages = {
        "without_docs": {"name": f"{prefix}-without-docs:local", "id": ""},
        "with_docs": {"name": f"{prefix}-with-docs:local", "id": ""},
    }
    for variant in ("without_docs", "with_docs"):
        _require_success("docker", [
            "build",
            "--target", variant,
            "--tag", images[variant]["name"],
            "--file", str(package_root / "docker" / "Dockerfile"),
            str(repository_root),
        ])
        status, stdout = _execute("docker", ["image", "inspect", "--format", "{{.Id}}", images[variant]["name"]],
                                  capture=True)
        if status != 0 or not stdout.strip():
            raise RuntimeError(f"Cannot inspect {images[variant]['name']}.")
        images[variant]["id"] = stdout.strip()
    return images


def _environment(name: str, value: str) -> List[str]:
    return ["--env", f"{name}={value}"]


def require_eval_auth_file(provider: str) -> str:
    """The host agent dir must contain auth.json with a credential for provider."""
    base = os.environ.get("PI_PY_AGENT_DIR") or str(Path.home() / ".pi-python")
    path = str(Path(base).resolve() / "auth.json")
    if not os.path.exists(path) or not os.path.isfile(path):
        raise RuntimeError(f"Eval authentication file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            credentials = json.load(handle)
    except ValueError as error:
        raise RuntimeError(f"Eval authentication file is invalid: {path}") from error
    if (not isinstance(credentials, dict) or provider not in credentials):
        raise RuntimeError(f"Eval authentication file has no credential for provider {provider}.")
    return path


def docker_args(context: DockerContext, variant: DocumentationVariant, output_directory: str,
                entrypoint_args: List[str]) -> List[str]:
    os.makedirs(output_directory, mode=0o700, exist_ok=True)
    os.chmod(output_directory, 0o700)
    args = [
        "run",
        "--rm",
        "--read-only",
        "--tmpfs", "/tmp:rw,exec,mode=1777",
        "--mount", f"type=bind,source={output_directory},target=/artifacts",
        *_environment("PI_EVAL_ARTIFACT_DIR", "/artifacts"),
        *_environment("PI_EVAL_RUNS_PER_VARIANT", str(context.runs_per_variant)),
        *_environment("PI_EVAL_SANDBOX_UID", "65532"),
        *_environment("PI_EVAL_SANDBOX_GID", "65532"),
        *_environment("PI_PROVIDER", context.provider),
        *_environment("PI_MODEL", context.model),
    ]
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        args.extend(_environment("PI_EVAL_ARTIFACT_UID", str(os.getuid())))
        args.extend(_environment("PI_EVAL_ARTIFACT_GID", str(os.getgid())))
    args.append("--mount")
    args.append(f"type=bind,source={context.auth_path},target=/run/pi-eval-secrets/auth.json,readonly")
    args.append(context.images[variant]["name"])
    args.extend(entrypoint_args)
    return args


def create_docker_context(images: BuiltImages, artifact_directory: str, auth_path: str,
                          provider: str, model: str, runs_per_variant: int) -> DockerContext:
    return DockerContext(images=images, artifact_directory=artifact_directory, auth_path=auth_path,
                         provider=provider, model=model, runs_per_variant=runs_per_variant)


def discover_cases(context: DockerContext, variant: DocumentationVariant, files: List[str],
                   vitest_args: List[str]) -> str:
    """Run discovery inside the variant container; return the report path."""
    output_directory = os.path.join(context.artifact_directory, "discovery", variant)
    args = docker_args(context, variant, output_directory,
                       ["--discover", "--project", "docs", *files, *vitest_args])
    status, _ = _execute("docker", args)
    if status != 0:
        raise RuntimeError(f"{variant} eval discovery failed.")
    return os.path.join(output_directory, "discovered-tests.json")


def task_directory_name(task: EvalTask) -> str:
    identity = json.dumps([task.eval_set, task.case_id, task.variant, task.model, task.run_number],
                          ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def run_task(context: DockerContext, task: EvalTask) -> Optional[str]:
    """Run one planned arm; return the report path when one was produced."""
    output_directory = os.path.join(context.artifact_directory, "tasks", task_directory_name(task))
    test_name = f"{task.eval_set} {task.case_id}"
    exact_name = "^" + re.sub(r"[.*+?^${}()|[\]\\]", lambda match: "\\" + match.group(0), test_name) + "$"
    args = docker_args(context, task.variant, output_directory,
                       ["--project", "docs", task.file, "--testNamePattern", exact_name])
    _execute("docker", args)
    report_path = os.path.join(output_directory, "vitest.json")
    return report_path if os.path.exists(report_path) else None
