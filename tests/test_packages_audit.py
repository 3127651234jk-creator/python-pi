"""Regression coverage for package installation and archive boundaries."""

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest

from pi.coding_agent.config import Configuration
from pi.coding_agent.packages import PackageManager, parse_source, safe_extract_tar


def test_git_install_uses_hostname(tmp_path, monkeypatch):
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    commands = []

    async def run_git(command, notify):
        commands.append(command)
        return "revision\n"

    monkeypatch.setattr(manager, "_run_git", run_git)
    name, version, root = asyncio.run(manager._install_git(
        parse_source("git:https://example.com/team/plugin.git#main"), lambda text: None))
    assert name == "plugin"
    assert version == "revision"
    assert root == tmp_path / "agent" / "git" / "example.com" / "team" / "plugin"
    assert commands[0][-2:] == ["https://example.com/team/plugin.git", str(root)]


@pytest.mark.parametrize("url", [
    "https:///plugin.git", "https://example.com/../plugin.git",
    "https://example.com/team/../../plugin.git", "https://example.com/team\\..\\plugin.git",
])
def test_git_install_rejects_unsafe_cache_paths(tmp_path, url):
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    with pytest.raises(ValueError):
        asyncio.run(manager._install_git(parse_source(url), lambda text: None))


def test_place_multiple_manifest_entries_preserves_every_resource(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"pi": {
        "extensions": ["first.py", "second.py"]}}), encoding="utf-8")
    for name in ("first.py", "second.py"):
        (root / name).write_text(name, encoding="utf-8")
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    placed = manager._place_files("plugin", root, lambda text: None)
    assert len(placed["extensions"]) == 2
    assert [Path(path).read_text(encoding="utf-8") for path in placed["extensions"]] == [
        "first.py", "second.py"]


@pytest.mark.parametrize("name", [
    "package/../escape.txt", "package/..\\escape.txt",
    "package/C:/escape.txt", "package/\\\\server\\share\\escape.txt",
])
def test_archive_rejects_platform_specific_escape_paths(tmp_path, name):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe tarball path"):
        safe_extract_tar(buffer.getvalue(), tmp_path / "target")


def test_archive_extracts_regular_npm_files(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("package/skills/example.md")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"ok"))
    safe_extract_tar(buffer.getvalue(), tmp_path / "target")
    assert (tmp_path / "target" / "skills" / "example.md").read_bytes() == b"ok"
