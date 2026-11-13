"""Mutations through directory aliases share the same queue."""

import asyncio
import os
import subprocess
import sys

import pytest

from pi.coding_agent.tools.common import with_file_mutation_queue


@pytest.mark.parametrize("existing", [False, True])
def test_mutations_through_directory_alias_are_serialized(tmp_path, existing):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(target)], check=True, capture_output=True)
    else:
        alias.symlink_to(target, target_is_directory=True)
    path = target / "file.txt"
    if existing:
        path.write_text("initial")

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        second_entered = False

        async def first():
            entered.set()
            await release.wait()
            path.write_text("first")

        async def second():
            nonlocal second_entered
            second_entered = True
            assert path.read_text() == "first"
            (alias / "file.txt").write_text("second")

        first_task = asyncio.create_task(with_file_mutation_queue(str(path), first))
        await entered.wait()
        second_task = asyncio.create_task(with_file_mutation_queue(str(alias / "file.txt"), second))
        try:
            await asyncio.sleep(0)
            assert not second_entered
        finally:
            release.set()
            await asyncio.gather(first_task, second_task)
        assert second_entered

    try:
        asyncio.run(scenario())
        assert path.read_text() == "second"
    finally:
        if sys.platform == "win32":
            os.rmdir(alias)
        else:
            alias.unlink()
