"""Regression coverage for the durable/coding-agent bug sweep."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from pi.agent.agent import Agent
from pi.agent.compaction import CompactionSettings, ContextCompactor
from pi.ai.models import Model, resolve_model
from pi.ai.providers import SimpleStreamOptions
from pi.ai.providers import openai_responses
from pi.ai.types import DoneEvent, ToolCall, TranscriptContext, UserMessage
from pi.coding_agent.config import Configuration
from pi.coding_agent.extensions import ExtensionManager
from pi.coding_agent.packages import PackageManager
from pi.durable.env import Context, LocalExecutionEnv
from pi.durable.errors import StorageRejected
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage
from pi.durable.tools.mutation_queue import _mutation_key


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


# -- durable environments ----------------------------------------------------


def test_remove_directory_requires_recursive(tmp_path):
    async def check():
        env = LocalExecutionEnv(cwd=str(tmp_path))
        (tmp_path / "tree" / "nested").mkdir(parents=True)
        marker = tmp_path / "tree" / "nested" / "file.txt"
        marker.write_text("keep", encoding="utf-8")

        refused = await env.remove("tree", None, Context())
        assert not refused.ok
        assert refused.error.code == "is_directory"
        assert marker.read_text(encoding="utf-8") == "keep"

        assert (await env.remove("tree", {"recursive": True}, Context())).ok
        assert not (tmp_path / "tree").exists()

    asyncio.run(check())


def test_missing_file_queue_key_resolves_parent_aliases(tmp_path):
    async def check():
        target = tmp_path / "target"
        target.mkdir()
        alias = tmp_path / "alias"
        if sys.platform == "win32":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(target)],
                           check=True, capture_output=True)
        else:
            alias.symlink_to(target, target_is_directory=True)
        try:
            env = LocalExecutionEnv(cwd=str(tmp_path))
            direct = await _mutation_key(env, str(target / "missing.txt"), Context())
            through_alias = await _mutation_key(env, str(alias / "missing.txt"), Context())
            assert direct == through_alias
        finally:
            if sys.platform == "win32":
                os.rmdir(alias)
            else:
                alias.unlink()

    asyncio.run(check())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows watcher backend")
def test_windows_watch_releases_event_handle(tmp_path):
    from pi.durable.env import native_backends

    backend = native_backends.WindowsWatchBackend(lambda directory, name: None)
    directory = tmp_path / "watched"
    directory.mkdir()
    try:
        backend.watch(str(directory))
        watch = backend._watches[str(directory)]
        assert watch.overlapped.hEvent
        backend.unwatch(str(directory))
        assert not watch.thread.is_alive()
        assert watch.overlapped.hEvent is None
    finally:
        backend.close()


# -- harness admission -------------------------------------------------------


def test_submissions_reject_unknown_conversation(tmp_path):
    async def check():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, make_model())
        harness.register_tools({})
        try:
            with pytest.raises(StorageRejected):
                await harness.submit_input(987_654, "hello")
            with pytest.raises(StorageRejected):
                await harness.submit_tool_task(987_654, "echo", {"value": "x"})
            with pytest.raises(StorageRejected):
                await harness.submit_compact_task(987_654)
            assert await harness.list_tasks() == []
            assert (await session.storage.scan_submissions({}, 10)).items == []
            assert (await session.storage.scan_conversations({}, 10)).items == []
        finally:
            await session.close()

    asyncio.run(check())


# -- package installs --------------------------------------------------------


def _write_package(root: Path, manifest: dict, files: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text(json.dumps({"pi": manifest}), encoding="utf-8")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _fake_source(manager: PackageManager, monkeypatch, name: str, version: str, root: Path) -> None:
    async def fake_install_git(parsed, notify):
        return name, version, root

    monkeypatch.setattr(manager, "_install_git", fake_install_git)


def test_empty_update_keeps_installed_files_and_record(tmp_path, monkeypatch):
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    v1 = tmp_path / "v1"
    _write_package(v1, {"extensions": ["plugin.py"]}, {"plugin.py": "v1"})
    _fake_source(manager, monkeypatch, "plugin", "v1", v1)
    installed = asyncio.run(manager.install("git:https://example.com/plugin.git"))
    path = Path(installed.files["extensions"][0])
    assert path.read_text(encoding="utf-8") == "v1"

    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "package.json").write_text("{}", encoding="utf-8")
    _fake_source(manager, monkeypatch, "plugin", "v2", v2)
    with pytest.raises(ValueError, match="no pi"):
        asyncio.run(manager.install("git:https://example.com/plugin.git"))

    assert path.read_text(encoding="utf-8") == "v1"
    record = manager.get("plugin")
    assert record is not None and record.version == "v1"


def test_failed_copy_keeps_installed_files(tmp_path, monkeypatch):
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    v1 = tmp_path / "v1"
    _write_package(v1, {"extensions": ["plugin.py"]}, {"plugin.py": "v1"})
    _fake_source(manager, monkeypatch, "plugin", "v1", v1)
    installed = asyncio.run(manager.install("git:https://example.com/plugin.git"))
    path = Path(installed.files["extensions"][0])

    v2 = tmp_path / "v2"
    _write_package(v2, {"extensions": ["plugin.py"]}, {"plugin.py": "v2"})
    _fake_source(manager, monkeypatch, "plugin", "v2", v2)
    real_copy = shutil.copy2

    def failing_copy(source, destination, *args, **kwargs):
        if str(v2) in str(source):
            raise OSError("disk full")
        return real_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr("pi.coding_agent.packages.shutil.copy2", failing_copy)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(manager.install("git:https://example.com/plugin.git"))

    assert path.read_text(encoding="utf-8") == "v1"
    record = manager.get("plugin")
    assert record is not None and record.version == "v1"


def test_update_replaces_kinds_absent_from_the_new_version(tmp_path, monkeypatch):
    manager = PackageManager(Configuration(str(tmp_path), str(tmp_path / "agent")))
    v1 = tmp_path / "v1"
    _write_package(v1, {"extensions": ["plugin.py"]}, {"plugin.py": "v1"})
    _fake_source(manager, monkeypatch, "plugin", "v1", v1)
    installed = asyncio.run(manager.install("git:https://example.com/plugin.git"))
    old_extension = Path(installed.files["extensions"][0])

    v2 = tmp_path / "v2"
    _write_package(v2, {"skills": ["skills"]}, {"skills/tool.md": "skill"})
    _fake_source(manager, monkeypatch, "plugin", "v2", v2)
    updated = asyncio.run(manager.install("git:https://example.com/plugin.git"))

    assert not old_extension.exists()
    assert not (tmp_path / "agent" / "extensions" / "plugin").exists()
    skill = Path(updated.files["skills"][0])
    assert (skill / "tool.md").read_text(encoding="utf-8") == "skill"
    assert manager.get("plugin").version == "v2"


# -- extensions --------------------------------------------------------------


def test_reload_keeps_the_ui_sink(tmp_path):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi" / "extensions"
        directory.mkdir(parents=True)
        (directory / "demo.py").write_text(
            "def setup(api):\n    api.register_command('ping', lambda args, ctx: None)\n",
            encoding="utf-8")
        agent = Agent(resolve_model("faux"))
        manager = ExtensionManager(config, agent)
        try:
            await manager.reload()
            requests = []

            async def sink(request):
                requests.append(request)
                return None

            manager.set_ui_sink(sink)
            await manager.reload()
            assert manager.extensions
            assert all(api.ui.sink is sink for api in manager.extensions)
            await manager.extensions[0].ui.notify("hello")
            assert requests[-1]["method"] == "notify"
        finally:
            await manager.close()

    asyncio.run(check())


# -- providers ---------------------------------------------------------------


@pytest.mark.parametrize("payload, expected", [
    ([
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "function_call", "call_id": "call", "name": "echo", "arguments": ""}},
        {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": ""},
    ], {}),
    ([
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "function_call", "call_id": "call", "name": "echo", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 1,
         "delta": '{"text":"hello"}'},
        {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": ""},
    ], {"text": "hello"}),
])
def test_empty_function_call_arguments_done(monkeypatch, payload, expected):
    async def send(self, request, **kwargs):
        items = [*payload, {"type": "response.completed",
                            "response": {"id": "r", "usage": {"input_tokens": 1, "output_tokens": 1}}}]
        content = "".join("data: " + json.dumps(item) + "\n\n" for item in items) + "data: [DONE]\n\n"
        return httpx.Response(200, request=request, content=content.encode())

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model = Model(id="test", name="test", api="openai-responses", provider="test",
                  base_url="https://example.test", api_key="key")

    async def check():
        context = TranscriptContext([UserMessage(content="hello", timestamp=0)])
        return [event async for event in openai_responses.stream_openai_responses(model, context)]

    events = asyncio.run(check())
    done = next(event for event in events if isinstance(event, DoneEvent))
    call = next(block for block in done.message.content if isinstance(block, ToolCall))
    assert call.arguments == expected
    assert done.message.stop_reason == "toolUse"


# -- storage and compaction --------------------------------------------------


def make_storage(backend, tmp_path):
    if backend == "memory":
        return MemoryStorage()
    if backend == "jsonl":
        return JsonlStorage(str(tmp_path / f"{backend}.jsonl"))
    return SqliteStorage(str(tmp_path / f"{backend}.db"))


@pytest.mark.parametrize("backend", ["memory", "jsonl", "sqlite"])
def test_non_positive_page_limit_is_rejected(backend, tmp_path):
    async def check():
        storage = make_storage(backend, tmp_path)
        try:
            for limit in (0, -1):
                with pytest.raises(StorageRejected):
                    await storage.scan_conversations({}, limit)
        finally:
            await storage.close()

    asyncio.run(check())


def test_zero_keep_recent_tokens_does_not_index_past_the_tail():
    async def check():
        compactor = ContextCompactor(CompactionSettings(keep_recent_tokens=0))
        model = resolve_model("faux")
        model.context_window = 200
        messages = [UserMessage(content="old " * 500, timestamp=0),
                    UserMessage(content="latest", timestamp=1)]

        async def stream(model, context, options):
            from pi.ai.types import AssistantMessage, TextContent, Usage

            yield DoneEvent(reason="stop", message=AssistantMessage(
                content=[TextContent(text="summary")], api="faux", provider="faux",
                model="faux", usage=Usage(), stop_reason="stop", timestamp=0))

        projected = await compactor.project(messages, model, SimpleStreamOptions(), stream)
        assert projected[-1] is messages[-1]

    asyncio.run(check())
