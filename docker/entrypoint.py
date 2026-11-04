"""Container entrypoint for the Python documentation evals.

Port of ``packages/evals/docker/entrypoint.ts``. Validates the image contents,
protects evaluator sources from the unprivileged eval user, stages the shared
credential, and either discovers eval cases or runs one selected case, writing
Vitest-schema JSON reports into ``/artifacts``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path("/repo")
PACKAGE_ROOT = REPO / "pi-python"
ARTIFACTS = Path("/artifacts")


def parse_id(name: str) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        raise RuntimeError(f"{name} must be a positive integer.") from None
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def assert_entries(path: Path, expected: List[str]) -> None:
    actual = sorted(os.listdir(path))
    if actual != sorted(expected):
        raise RuntimeError(f"Unexpected eval image contents in {path}: {', '.join(actual)}")


def assert_workspace() -> None:
    assert_entries(REPO, [".venv", "pi-python"])
    assert_entries(PACKAGE_ROOT, ["docker", "evals", "pyproject.toml"])
    installed = site_packages() / "pi"
    if not (installed / "__init__.py").exists():
        raise RuntimeError("Installed pi package is missing __init__.py.")
    if not (installed / "evals").is_dir():
        raise RuntimeError("Installed pi package is missing evals.")
    variant = os.environ.get("PI_EVAL_VARIANT")
    if variant == "without_docs":
        for name in ("README.md", "CHANGELOG.md", "docs", "examples"):
            if (site_packages() / name).exists():
                raise RuntimeError(f"without_docs contains pi documentation: {name}")
        return
    if variant != "with_docs":
        raise RuntimeError("Eval image has no valid PI_EVAL_VARIANT.")
    readme = site_packages() / "README.md"
    if not readme.exists() or readme.stat().st_size == 0:
        raise RuntimeError(f"Missing documentation: {readme}")


def assert_root_only(path: Path) -> None:
    stats = os.stat(path)
    if stats.st_uid != 0 or stats.st_mode & 0o077:
        raise RuntimeError(f"Evaluator source must be readable only by root: {path}")


def assert_sandbox_cannot_read(path: Path) -> None:
    if os.path.isdir(path):
        probe = f"import os; os.listdir({str(path)!r})"
    else:
        probe = f"open({str(path)!r}, 'rb').close()"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        user=parse_id("PI_EVAL_SANDBOX_UID"),
        group=parse_id("PI_EVAL_SANDBOX_GID"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode == 0:
        raise RuntimeError(f"The model-facing user can read evaluator source: {path}")


def chown_tree(path: Path, uid: int, gid: int) -> None:
    stats = os.lstat(path)
    if stat.S_ISDIR(stats.st_mode):
        for entry in os.listdir(path):
            chown_tree(path / entry, uid, gid)
    os.chown(path, uid, gid, follow_symlinks=False)


def parse_arguments(argv: List[str]) -> Tuple[bool, Optional[str], List[Path]]:
    """Split --discover / --project / --testNamePattern from eval file paths."""
    discover = "--discover" in argv
    remaining: List[str] = []
    test_name_pattern: Optional[str] = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--discover":
            index += 1
            continue
        if argument == "--project":
            index += 2
            continue
        if argument == "--testNamePattern":
            if index + 1 >= len(argv):
                raise RuntimeError("Missing value for --testNamePattern.")
            test_name_pattern = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--testNamePattern="):
            test_name_pattern = argument[len("--testNamePattern="):]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    eval_files: List[Path] = []
    for argument in remaining:
        if not argument.endswith(".eval.py"):
            continue
        path = Path(argument)
        if not path.is_absolute():
            path = PACKAGE_ROOT / path
        eval_files.append(path)
    if not eval_files:
        raise RuntimeError("The container runner requires an explicit eval file.")
    return discover, test_name_pattern, eval_files


def main(argv: List[str]) -> int:
    assert_workspace()
    eval_root = PACKAGE_ROOT / "evals"
    protected = [eval_root, PACKAGE_ROOT / "docker", PACKAGE_ROOT / "pyproject.toml",
                 site_packages() / "pi" / "evals"]
    for path in protected:
        assert_root_only(path)
        assert_sandbox_cannot_read(path)

    agent_dir = Path("/tmp/pi-eval-host-agent")
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_source = Path("/run/pi-eval-secrets/auth.json")
    if auth_source.exists():
        shutil.copyfile(auth_source, agent_dir / "auth.json")
    sandbox_uid = parse_id("PI_EVAL_SANDBOX_UID")
    sandbox_gid = parse_id("PI_EVAL_SANDBOX_GID")
    chown_tree(agent_dir, sandbox_uid, sandbox_gid)
    chown_tree(ARTIFACTS, sandbox_uid, sandbox_gid)
    os.environ["HOME"] = "/tmp/pi-eval-bootstrap"
    os.environ["USERPROFILE"] = os.environ["HOME"]
    os.environ["PI_PY_AGENT_DIR"] = str(agent_dir)
    os.environ["PI_EVAL_CONTAINER"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.umask(0o022)

    discover, test_name_pattern, eval_files = parse_arguments(argv)
    for eval_file in eval_files:
        assert_root_only(eval_file)
        assert_sandbox_cannot_read(eval_file)

    from pi.evals.framework import discover_eval_files, run_eval_file

    status = 0
    if discover:
        records = discover_eval_files([str(path) for path in eval_files])
        (ARTIFACTS / "discovered-tests.json").write_text(
            json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        if len(eval_files) != 1:
            raise RuntimeError("The container runner executes exactly one eval file per task.")
        report = asyncio.run(run_eval_file(str(eval_files[0]), test_name_pattern))
        (ARTIFACTS / "vitest.json").write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
        status = 0 if report["success"] else 1

    if os.environ.get("PI_EVAL_ARTIFACT_UID") and os.environ.get("PI_EVAL_ARTIFACT_GID"):
        chown_tree(ARTIFACTS, parse_id("PI_EVAL_ARTIFACT_UID"), parse_id("PI_EVAL_ARTIFACT_GID"))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
