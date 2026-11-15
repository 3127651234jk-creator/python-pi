"""ExecutionEnv conformance suite runs against the local environment.

Each case runs as its own pytest test against a ``LocalExecutionEnv`` whose
``cwd`` is a fresh, empty, writable directory.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.env import Context, LocalExecutionEnv
from pi.durable.testing.types import ConformanceAssertions, EnvConformanceOptions


def _find_shell():
    for name in ("bash", "sh"):
        found = shutil.which(name)
        if found:
            return [found, "-c"]
    # Git for Windows ships bash next to git.exe, off the PATH.
    git = shutil.which("git")
    if git:
        root = os.path.dirname(os.path.dirname(os.path.abspath(git)))
        for candidate in (
            os.path.join(root, "bin", "bash.exe"),
            os.path.join(root, "usr", "bin", "bash.exe"),
        ):
            if os.path.exists(candidate):
                return [candidate, "-c"]
    pytest.skip("no POSIX shell available for env conformance")


def _case_names():
    from pi.durable.testing import create_env_conformance

    options = EnvConformanceOptions(
        assertions=ConformanceAssertions(), with_env=None, shell=["sh", "-c"], symlinks=True
    )
    return [case.name for case in create_env_conformance(options)]


CASE_NAMES = _case_names()


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_env_conformance(case_name, tmp_path):
    shell = _find_shell()
    symlinks = os.name != "nt"  # Windows ``ln -s`` needs privileges; skip the symlink cases there.
    if os.name == "nt" and (
        case_name.startswith("binary reader follows symlinks")
        or case_name.startswith("watch reports changes to the file a watched symbolic link")
    ):
        pytest.skip("symbolic links need privileges on Windows")

    async def main():
        env = LocalExecutionEnv(cwd=str(tmp_path))
        try:

            async def with_env(use):
                await use(env)

            from pi.durable.testing import create_env_conformance

            options = EnvConformanceOptions(
                assertions=ConformanceAssertions(), with_env=with_env, shell=shell, symlinks=symlinks
            )
            cases = create_env_conformance(options)
            target = next(candidate for candidate in cases if candidate.name == case_name)
            await asyncio.wait_for(target.run(), timeout=candidate_timeout(target))
        finally:
            await env.cleanup(Context())

    asyncio.run(main())


def candidate_timeout(case):
    return case.timeout_seconds or 30.0


def test_cleanup_kills_a_command_still_starting(tmp_path, monkeypatch):
    async def main():
        env = LocalExecutionEnv(cwd=str(tmp_path))
        original = asyncio.create_subprocess_exec
        spawn_started = asyncio.Event()
        allow_spawn = asyncio.Event()

        async def delayed_spawn(*args, **kwargs):
            if args[0] == sys.executable:
                spawn_started.set()
                await allow_spawn.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        task = asyncio.create_task(env.exec(command, None, Context()))
        await spawn_started.wait()
        await env.cleanup(Context())
        allow_spawn.set()
        result = await asyncio.wait_for(task, timeout=10)
        assert result.ok
        assert result.value.exit_code != 0

    asyncio.run(main())
