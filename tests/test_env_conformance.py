"""ExecutionEnv conformance suite runs against ``RemoteExecutionEnv``.

Like ``packages/env``'s ``test/conformance.test.ts``: each case runs against a
``RemoteExecutionEnv`` whose daemon (``pi.env.daemon``) is started over the
connection's command on the same machine, so the suite checks the whole stack:
client path and error rules, wire protocol, and the daemon's system calls.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

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
        from pi.durable.testing import create_env_conformance
        from pi.env import Connection, RemoteExecutionEnv

        connection = Connection([sys.executable, "-m", "pi.env.daemon"])
        environment = RemoteExecutionEnv(connection=connection, id=f"test-env:{case_name}", cwd=str(tmp_path))
        try:

            async def with_env(use):
                await use(environment)

            options = EnvConformanceOptions(
                assertions=ConformanceAssertions(), with_env=with_env, shell=shell, symlinks=symlinks
            )
            cases = create_env_conformance(options)
            target = next(candidate for candidate in cases if candidate.name == case_name)
            await asyncio.wait_for(target.run(), timeout=target.timeout_seconds or 60.0)
        finally:
            await environment.cleanup(None)
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())
