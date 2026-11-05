"""Tool-level tests for the durable read/write/edit/bash tools.

Runs each tool through a small in-process ToolExecutionApi backed by a
LocalExecutionEnv, covering truncation diagnostics, edit matching rules, and
bash output bounding beyond the conformance cases.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.env import Context, LocalExecutionEnv
from pi.durable.output import OutputBuffer, OutputLimits
from pi.durable.tools import (
    ToolDiagnostic,
    ToolExecutionApi,
    ToolExecutionResult,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)


class RecordingApi(ToolExecutionApi):
    """Records running output in a bounded buffer, like the harness would."""

    def __init__(self, env: LocalExecutionEnv, retain: str = "tail") -> None:
        self.env = env
        self.buffer = OutputBuffer(OutputLimits(max_bytes=50 * 1024, max_lines=2000, retain=retain))
        self.output_window = None
        self.diagnostics: list = []

    def output(self, chunk, skipped=None) -> None:
        self.buffer.push(chunk, skipped)

    def diagnostic(self, diagnostic: ToolDiagnostic) -> None:
        self.diagnostics.append(diagnostic)


@pytest.fixture()
def env(tmp_path):
    return LocalExecutionEnv(cwd=str(tmp_path))


def run(coro):
    return asyncio.run(coro)


def _text(result: ToolExecutionResult) -> str:
    if result.content is not None:
        return "".join(block.get("text", "") for block in result.content)
    return ""


def test_read_write_roundtrip(env):
    write = create_write_tool()
    read = create_read_tool()
    api = RecordingApi(env)
    result = run(write.execute({"path": "hello.txt", "content": "one\ntwo\n"}, api, Context()))
    assert _text(result) == "Successfully wrote to hello.txt"
    assert os.path.exists(os.path.join(env.cwd, "hello.txt"))
    result = run(read.execute({"path": "hello.txt"}, api, Context()))
    assert _text(result) == "one\ntwo\n"
    assert result.diagnostics in (None, [])


def test_write_creates_parent_directories(env):
    write = create_write_tool()
    api = RecordingApi(env)
    result = run(write.execute({"path": "a/b/c/file.txt", "content": "x"}, api, Context()))
    assert _text(result) == "Successfully wrote to a/b/c/file.txt"
    assert os.path.exists(os.path.join(env.cwd, "a", "b", "c", "file.txt"))


def test_read_truncation_reports_offset(env):
    content = "".join(f"line-{index}\n" for index in range(100))
    write, read = create_write_tool(), create_read_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "big.txt", "content": content}, api, Context()))
    result = run(read.execute({"path": "big.txt", "limit": 10}, api, Context()))
    # The selection ends mid-file, so the last line's newline is excluded, like the TS slice.
    assert _text(result) == "\n".join(f"line-{index}" for index in range(10))
    assert result.diagnostics and "91 more lines" in result.diagnostics[0].message
    assert "Use offset=11" in result.diagnostics[0].message


def test_read_truncate_head_limit(env):
    content = "".join(f"line-{index}\n" for index in range(3000))
    write, read = create_write_tool(), create_read_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "huge.txt", "content": content}, api, Context()))
    result = run(read.execute({"path": "huge.txt"}, api, Context()))
    assert result.details["truncation"]["outputLines"] == 2000
    assert result.diagnostics and "Showing lines 1-2000 of 3001" in result.diagnostics[0].message
    assert "Use offset=2001" in result.diagnostics[0].message


def test_read_offset_continuation(env):
    content = "".join(f"line-{index}\n" for index in range(50))
    write, read = create_write_tool(), create_read_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "big.txt", "content": content}, api, Context()))
    result = run(read.execute({"path": "big.txt", "offset": 45, "limit": 10}, api, Context()))
    # The selection runs past the last line, so it ends at the end of the file, newline included.
    assert _text(result) == "".join(f"line-{index}\n" for index in range(44, 50))
    # The selection reaches the end of the file, so there is no continuation remark.
    assert result.diagnostics == []


def test_read_offset_beyond_end(env):
    write, read = create_write_tool(), create_read_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "small.txt", "content": "one\ntwo\n"}, api, Context()))
    with pytest.raises(ValueError, match="beyond end of file"):
        run(read.execute({"path": "small.txt", "offset": 99}, api, Context()))


def test_read_refuses_images(env):
    # Minimal PNG: signature + IHDR length and type; detected, then refused with a diagnostic.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 40
    read = create_read_tool()
    api = RecordingApi(env)
    with open(os.path.join(env.cwd, "image.png"), "wb") as handle:
        handle.write(png)
    result = run(read.execute({"path": "image.png"}, api, Context()))
    assert result.is_error is True
    assert result.diagnostics[0].code == "unsupported_image"


def test_edit_unique_and_overlapping(env):
    write, edit = create_write_tool(), create_edit_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "code.txt", "content": "alpha\nbeta\ngamma\n"}, api, Context()))
    result = run(
        edit.execute(
            {
                "path": "code.txt",
                "edits": [{"oldText": "beta", "newText": "BETA"}, {"oldText": "gamma", "newText": "GAMMA"}],
            },
            api,
            Context(),
        )
    )
    assert "Successfully replaced 2 block(s)" in _text(result)
    with open(os.path.join(env.cwd, "code.txt"), encoding="utf-8") as handle:
        assert handle.read() == "alpha\nBETA\nGAMMA\n"
    assert "firstChangedLine" in result.details
    assert result.details["firstChangedLine"] == 2


def test_edit_rejects_duplicates(env):
    write, edit = create_write_tool(), create_edit_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "dup.txt", "content": "same\nsame\n"}, api, Context()))
    with pytest.raises(ValueError, match="must be unique"):
        run(edit.execute({"path": "dup.txt", "edits": [{"oldText": "same", "newText": "x"}]}, api, Context()))


def test_edit_rejects_overlap(env):
    write, edit = create_write_tool(), create_edit_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "ov.txt", "content": "abcdefgh\n"}, api, Context()))
    with pytest.raises(ValueError, match="overlap"):
        run(
            edit.execute(
                {
                    "path": "ov.txt",
                    "edits": [
                        {"oldText": "abcd", "newText": "x"},
                        {"oldText": "cdef", "newText": "y"},
                    ],
                },
                api,
                Context(),
            )
        )


def test_edit_fuzzy_match_preserves_lines(env):
    write, edit = create_write_tool(), create_edit_tool()
    api = RecordingApi(env)
    run(write.execute({"path": "f.txt", "content": "café — keep\nplain\n"}, api, Context()))
    # oldText written with an ASCII hyphen matches the em dash via fuzzy normalization.
    result = run(
        edit.execute({"path": "f.txt", "edits": [{"oldText": "café - keep", "newText": "changed"}]}, api, Context())
    )
    with open(os.path.join(env.cwd, "f.txt"), encoding="utf-8") as handle:
        assert handle.read() == "changed\nplain\n"


def test_edit_rejects_empty_edits(env):
    edit = create_edit_tool()
    api = RecordingApi(env)
    with pytest.raises(ValueError, match="at least one replacement"):
        run(edit.execute({"path": "x.txt", "edits": []}, api, Context()))


def test_edit_prepare_arguments_repairs_shapes(env):
    prepare = create_edit_tool().prepare_arguments
    repaired = prepare({"path": "a.txt", "edits": '[{"oldText": "x", "newText": "y"}]'})
    assert repaired["edits"] == [{"oldText": "x", "newText": "y"}]
    repaired = prepare({"path": "a.txt", "edits": {"oldText": "x", "newText": "y"}})
    assert repaired["edits"] == [{"oldText": "x", "newText": "y"}]
    repaired = prepare({"path": "a.txt", "oldText": "x", "newText": "y"})
    assert repaired["edits"] == [{"oldText": "x", "newText": "y"}]


def test_bash_exit_code_and_output(env):
    bash = create_bash_tool()
    api = RecordingApi(env)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        run(bash.execute({"command": "printf hello; exit 3"}, api, Context()))
    assert api.buffer.snapshot().text.startswith("hello")


def test_bash_timeout(env):
    bash = create_bash_tool()
    api = RecordingApi(env)
    with pytest.raises(TimeoutError, match="timed out"):
        run(bash.execute({"command": "sleep 2", "timeout": 0.2}, api, Context()))


def test_bash_success(env):
    bash = create_bash_tool()
    api = RecordingApi(env)
    result = run(bash.execute({"command": "echo done"}, api, Context()))
    assert result.content is None or result.content == []
    assert "done" in api.buffer.snapshot().text


def test_bash_windowed_output_counts_skips(env):
    from pi.durable.env import ShellOutputWindow

    bash = create_bash_tool()
    api = RecordingApi(env)
    api.output_window = ShellOutputWindow(
        max_bytes=200, max_lines=5, min_interval_ms=0, bytes_per_second=1_000_000_000
    )
    script = "i=0; while [ $i -lt 500 ]; do echo line-$i; i=$((i+1)); done"
    run(bash.execute({"command": script}, api, Context()))
    snapshot = api.buffer.snapshot()
    assert snapshot.text.endswith("line-497\nline-498\nline-499\n")
    assert snapshot.dropped_bytes > 0


def test_bash_spill_reports_full_output(env, tmp_path):
    bash = create_bash_tool()
    api = RecordingApi(env)
    # 9000 lines at ~9 bytes each crosses the 50KB spill threshold.
    script = "i=0; while [ $i -lt 9000 ]; do echo line-$i; i=$((i+1)); done"
    run(bash.execute({"command": script}, api, Context()))
    spill = [d for d in api.diagnostics if d.code == "full_output"]
    assert spill and os.path.exists(spill[0].message.split("Full output: ")[1])


def test_read_bom_and_crlf_roundtrip(env):
    write, edit = create_write_tool(), create_edit_tool()
    api = RecordingApi(env)
    with open(os.path.join(env.cwd, "win.txt"), "wb") as handle:
        handle.write(b"\xef\xbb\xbfcrlf line one\r\ncrlf line two\r\n")
    result = run(
        edit.execute(
            {"path": "win.txt", "edits": [{"oldText": "crlf line one", "newText": "changed one"}]},
            api,
            Context(),
        )
    )
    with open(os.path.join(env.cwd, "win.txt"), "rb") as handle:
        data = handle.read()
    assert data == b"\xef\xbb\xbfchanged one\r\ncrlf line two\r\n"
