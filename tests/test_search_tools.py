import asyncio
import shutil

import pytest

from pi.coding_agent.tools import create_default_tools, tool_prompt_contributions
from pi.coding_agent.tools.powershell import create_powershell_tool
from pi.util import AbortError, AbortSignal


def execute(root, name, params, signal=None):
    tool = next(tool for tool in create_default_tools(str(root)) if tool.name == name)
    return asyncio.run(tool.execute("test", params, signal))


@pytest.fixture
def files(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("secret match", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("before\nHello [world]\nafter\nHELLO again\n", encoding="utf-8")
    (tmp_path / "src" / "b.txt").write_text("hello text\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("hello hidden\n", encoding="utf-8")
    return tmp_path


def test_registration_and_prompt(tmp_path):
    tools = create_default_tools(str(tmp_path))
    names = [tool.name for tool in tools]
    assert len(set(names)) == len(names) == 8
    snippets, _ = tool_prompt_contributions(tools)
    assert set(names) == set(snippets)


def test_ls_sorted_hidden_directories_and_limit(files):
    result = execute(files, "ls", {})
    assert ".hidden.py" in result.content[0].text
    assert "src/" in result.content[0].text
    limited = execute(files, "ls", {"limit": 1})
    assert limited.details["resultLimitReached"]
    with pytest.raises(ValueError):
        execute(files, "ls", {"limit": -1})
    with pytest.raises(NotADirectoryError):
        execute(files, "ls", {"path": "ignored.txt"})


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep unavailable")
def test_find_globs_relative_paths_and_ignores(files):
    text = execute(files, "find", {"pattern": "*.py"}).content[0].text.replace("\\", "/")
    assert "src/a.py" in text and ".hidden.py" in text
    assert "ignored.txt" not in execute(files, "find", {"pattern": "*"}).content[0].text
    assert "src/a.py" in execute(files, "find", {"pattern": "src/**/*.py"}).content[0].text.replace("\\", "/")
    assert "a.py" in execute(files, "find", {"path": "src", "pattern": "*.py"}).content[0].text
    assert execute(files, "find", {"pattern": "*", "limit": 1}).details["resultLimitReached"]


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep unavailable")
def test_grep_literal_case_context_glob_and_limit(files):
    (files / ".gitignore").write_text("ignored.txt\nignored.py\n", encoding="utf-8")
    (files / "ignored.py").write_text("secret match", encoding="utf-8")
    result = execute(files, "grep", {"pattern": "hello", "ignoreCase": True, "glob": "*.py", "limit": 1})
    assert result.details["resultLimitReached"]
    literal = execute(files, "grep", {"pattern": "Hello [world]", "literal": True, "context": 1})
    assert ":2:Hello [world]" in literal.content[0].text
    assert "before" in literal.content[0].text and "after" in literal.content[0].text
    assert "No matches" in execute(files, "grep", {"pattern": "secret"}).content[0].text
    assert "No matches" in execute(files, "grep", {"pattern": "secret", "glob": "*.py"}).content[0].text
    assert "No matches" in execute(files, "grep", {"pattern": "-not-an-option"}).content[0].text
    with pytest.raises(RuntimeError):
        execute(files, "grep", {"pattern": "["})
    with pytest.raises(FileNotFoundError):
        execute(files, "grep", {"pattern": "x", "path": "missing"})


def test_abort_and_missing_executables(files, monkeypatch):
    async def abort_check():
        signal = AbortSignal()
        signal.abort()
        for name, params in (("ls", {}), ("find", {"pattern": "*"}), ("grep", {"pattern": "x"})):
            tool = next(t for t in create_default_tools(str(files)) if t.name == name)
            with pytest.raises(AbortError):
                await tool.execute("test", params, signal)
    asyncio.run(abort_check())
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ripgrep"):
        execute(files, "find", {"pattern": "*"})
    with pytest.raises(FileNotFoundError, match="PowerShell"):
        execute(files, "powershell", {"command": "echo hello"})


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="PowerShell unavailable")
def test_powershell_unicode_exit_and_timeout(tmp_path):
    assert "你好" in execute(tmp_path, "powershell", {"command": "Write-Output '你好'"}).content[0].text
    assert execute(tmp_path, "powershell", {"command": "exit 7"}).is_error
    with pytest.raises(RuntimeError, match="timed out"):
        execute(tmp_path, "powershell", {"command": "Start-Sleep -Seconds 10", "timeout": 0.3})


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep unavailable")
def test_grep_output_byte_limit_and_long_lines(tmp_path):
    (tmp_path / "big.txt").write_text(("needle " + "x" * 300 + "\n") * 400, encoding="utf-8")
    result = execute(tmp_path, "grep", {"pattern": "needle", "limit": 1000})
    assert result.details["byteLimitReached"]
    assert len(result.content[0].text.encode("utf-8")) < 52000
    (tmp_path / "long.txt").write_text("needle " + "x" * 100000, encoding="utf-8")
    long = execute(tmp_path, "grep", {"pattern": "needle", "path": "long.txt"})
    assert len(long.content[0].text) < 2000


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="PowerShell unavailable")
def test_powershell_tail_truncation_and_abort(tmp_path):
    result = execute(tmp_path, "powershell", {"command": "1..2100 | ForEach-Object { Write-Output ('line' + $_) }"})
    text = result.content[0].text
    assert "line2100" in text
    assert "line1\n" not in text
    assert "Showing lines" in text
    assert result.details["fullOutputPath"]

    async def check():
        signal = AbortSignal()
        tool = create_powershell_tool(str(tmp_path))
        task = asyncio.create_task(tool.execute("test", {"command": "Start-Sleep -Seconds 10"}, signal))
        await asyncio.sleep(0.3)
        signal.abort()
        with pytest.raises(RuntimeError, match="aborted"):
            await asyncio.wait_for(task, 5)
    asyncio.run(check())
