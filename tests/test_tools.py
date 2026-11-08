"""Tool tests: read, write, edit, bash against a temp directory."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.coding_agent.tools import create_default_tools
from pi.coding_agent.tools import bash as bash_module
from pi.coding_agent.tools.edit import prepare_edit_arguments
from pi.coding_agent.tools.truncate import truncate_head, truncate_tail


@pytest.fixture()
def tmpcwd(tmp_path):
    return str(tmp_path)


def get_tool(cwd, name):
    tools = {t.name: t for t in create_default_tools(cwd)}
    return tools[name]


def test_write_then_read(tmp_path):
    cwd = str(tmp_path)

    async def scenario():
        write = get_tool(cwd, "write")
        read = get_tool(cwd, "read")
        await write.execute("t1", {"path": "sub/dir/hello.txt", "content": "line1\nline2\n"})
        result = await read.execute("t2", {"path": "sub/dir/hello.txt"})
        assert result.content[0].text == "line1\nline2\n"

    asyncio.run(scenario())


def test_read_truncation_notice(tmp_path):
    cwd = str(tmp_path)
    lines = [f"line {i}" for i in range(50)]
    (tmp_path / "big.txt").write_text("\n".join(lines), encoding="utf-8")

    async def scenario():
        read = get_tool(cwd, "read")
        # A user-supplied limit stops early with a continuation notice.
        result = await read.execute("t1", {"path": "big.txt", "limit": 10})
        text = result.content[0].text
        assert "[40 more lines in file. Use offset=11 to continue.]" in text

        result2 = await read.execute("t2", {"path": "big.txt", "offset": 11, "limit": 10})
        assert "line 10" in result2.content[0].text
        assert "line 19" in result2.content[0].text
        assert "Use offset=21" in result2.content[0].text

        # Without a limit, the head truncation limit kicks in only for huge files;
        # 50 lines is under the 2000-line default, so no notice appears.
        result3 = await read.execute("t3", {"path": "big.txt"})
        assert "line 49" in result3.content[0].text

    asyncio.run(scenario())


def test_edit_single_and_multi(tmp_path):
    cwd = str(tmp_path)
    (tmp_path / "code.py").write_text("def a():\n    return 1\n\ndef b():\n    return 2\n", encoding="utf-8")

    async def scenario():
        edit = get_tool(cwd, "edit")
        read = get_tool(cwd, "read")

        result = await edit.execute(
            "t1",
            {
                "path": "code.py",
                "edits": [
                    {"oldText": "return 1", "newText": "return 10"},
                    {"oldText": "def b():", "newText": "def bee():"},
                ],
            },
        )
        assert not result.is_error
        content = (tmp_path / "code.py").read_text(encoding="utf-8")
        assert "return 10" in content
        assert "def bee():" in content
        assert result.details["diff"]

        # Verify the file content via the read tool round-trip.
        read_result = await read.execute("t2", {"path": "code.py"})
        assert "def bee():" in read_result.content[0].text

    asyncio.run(scenario())


def test_edit_rejects_duplicate_match(tmp_path):
    cwd = str(tmp_path)
    (tmp_path / "dup.txt").write_text("same\nsame\n", encoding="utf-8")

    async def scenario():
        edit = get_tool(cwd, "edit")
        with pytest.raises(ValueError, match="unique"):
            await edit.execute("t1", {"path": "dup.txt", "edits": [{"oldText": "same", "newText": "x"}]})

    asyncio.run(scenario())


def test_edit_rejects_not_found(tmp_path):
    cwd = str(tmp_path)
    (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")

    async def scenario():
        edit = get_tool(cwd, "edit")
        with pytest.raises(ValueError, match="Could not find"):
            await edit.execute("t1", {"path": "f.txt", "edits": [{"oldText": "nope", "newText": "x"}]})

    asyncio.run(scenario())


def test_edit_prepare_arguments_shims():
    # edits as JSON string
    prepared = prepare_edit_arguments({"path": "p", "edits": '[{"oldText": "a", "newText": "b"}]'})
    assert prepared["edits"] == [{"oldText": "a", "newText": "b"}]
    # single edit object
    prepared = prepare_edit_arguments({"path": "p", "edits": {"oldText": "a", "newText": "b"}})
    assert prepared["edits"] == [{"oldText": "a", "newText": "b"}]
    # legacy top-level oldText/newText
    prepared = prepare_edit_arguments({"path": "p", "oldText": "a", "newText": "b"})
    assert prepared["edits"] == [{"oldText": "a", "newText": "b"}]


def test_edit_preserves_crlf(tmp_path):
    cwd = str(tmp_path)
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nb\r\n")

    async def scenario():
        edit = get_tool(cwd, "edit")
        await edit.execute("t1", {"path": "crlf.txt", "edits": [{"oldText": "b", "newText": "bee"}]})
        data = (tmp_path / "crlf.txt").read_bytes()
        assert data == b"a\r\nbee\r\n"

    asyncio.run(scenario())


def test_bash_echo(tmp_path):
    cwd = str(tmp_path)

    async def scenario():
        bash = get_tool(cwd, "bash")
        result = await bash.execute("t1", {"command": "echo hello-pi"})
        text = result.content[0].text
        assert "hello-pi" in text
        assert not result.is_error

    asyncio.run(scenario())


def test_bash_nonzero_exit_is_error(tmp_path):
    cwd = str(tmp_path)

    async def scenario():
        bash = get_tool(cwd, "bash")
        if sys.platform == "win32":
            command = "exit 3"
        else:
            command = "exit 3"
        result = await bash.execute("t1", {"command": command})
        assert result.is_error
        assert "exited with code 3" in result.content[0].text

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell selection")
def test_bash_prefers_git_bash_when_installed():
    git_bash = bash_module._find_windows_bash()
    if git_bash is None:
        pytest.skip("Git Bash unavailable")
    argv, _ = bash_module._shell_command("echo hi")
    assert argv[0] == git_bash
    assert argv[1] == "-c"
    assert "bash" in bash_module.BASH_PROMPT_SNIPPET


def test_bash_timeout(tmp_path):
    cwd = str(tmp_path)

    async def scenario():
        bash = get_tool(cwd, "bash")
        # "sleep" exists both in bash and as a PowerShell alias for Start-Sleep.
        with pytest.raises(RuntimeError, match="timed out"):
            await bash.execute("t1", {"command": "sleep 5", "timeout": 1})

    asyncio.run(scenario())


def test_truncate_head_and_tail():
    head = truncate_head("\n".join(f"l{i}" for i in range(3000)))
    assert head.truncated and head.truncated_by == "lines"
    assert head.output_lines == 2000

    tail = truncate_tail("\n".join(f"l{i}" for i in range(3000)))
    assert tail.truncated and tail.output_lines == 2000
    assert tail.content.splitlines()[0] == "l1000"

    byte_bound = truncate_head("x" * 100 + "\n" + "y" * 100, max_bytes=150)
    assert byte_bound.truncated and byte_bound.truncated_by == "bytes"


def test_import_does_not_spin_up_an_event_loop():
    """Instantiating asyncio primitives at module import time creates (and
    leaks) an event loop on Python 3.9; the mutation-queue guards are created
    lazily on first use instead."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = (
        "import asyncio, sys\n"
        f"sys.path.insert(0, {root!r})\n"
        "import pi.coding_agent.tools.common, pi.durable.tools.mutation_queue\n"
        "policy = asyncio.get_event_loop_policy()\n"
        "loop = getattr(getattr(policy, '_local', None), '_loop', None)\n"
        "assert loop is None, f'import created an event loop: {loop!r}'\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
