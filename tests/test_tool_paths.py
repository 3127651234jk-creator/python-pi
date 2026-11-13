"""Tool path normalization matches the TypeScript client."""

import asyncio
import os
import sys

import pytest

from pi.coding_agent.tools.common import resolve_to_cwd
from pi.coding_agent.tools import create_default_tools


@pytest.mark.parametrize("path", ["folder/file.txt", "@folder/file.txt", "folder/../folder/file.txt"])
def test_relative_paths(tmp_path, path):
    assert resolve_to_cwd(path, str(tmp_path)) == str(tmp_path / "folder" / "file.txt")


@pytest.mark.parametrize("suffix", ["", "/folder/file.txt"])
def test_home_paths(tmp_path, suffix):
    assert resolve_to_cwd("@~" + suffix, str(tmp_path)) == os.path.normpath(os.path.expanduser("~") + suffix)


def test_file_url_and_unicode_spaces(tmp_path):
    path = tmp_path / "some file.txt"
    assert resolve_to_cwd("@" + path.as_uri(), str(tmp_path)) == str(path)
    assert resolve_to_cwd("some\u202ffile.txt", str(tmp_path)) == str(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell paths")
@pytest.mark.parametrize("path", ["/d/work/x", "/mnt/d/work/x", "/cygdrive/d/work/x", "@/d/work/x", "file:///D:/work/x"])
def test_shell_drive_paths_do_not_use_current_drive(path):
    assert resolve_to_cwd(path, r"C:\other") == r"D:\work\x"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths")
def test_windows_unc_and_drive_root():
    assert resolve_to_cwd("/d", r"C:\other") == "D:\\"
    assert resolve_to_cwd("//server/share/file", r"C:\other") == r"\\server\share\file"
    assert resolve_to_cwd("file://server/share/file", r"C:\other") == r"\\server\share\file"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_posix_drive_like_paths_are_preserved():
    assert resolve_to_cwd("/d/work/x", "/other") == "/d/work/x"


def test_normalized_paths_write_and_read_same_file(tmp_path):
    tools = {tool.name: tool for tool in create_default_tools(str(tmp_path))}
    path = tmp_path / "some file.txt"

    async def scenario():
        await tools["write"].execute("write", {"path": "@" + path.as_uri(), "content": "expected"})
        result = await tools["read"].execute("read", {"path": "@some\u202ffile.txt"})
        assert result.content[0].text == "expected"

    asyncio.run(scenario())
    assert path.read_text() == "expected"
