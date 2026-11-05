"""ExecutionEnv conformance suite (port of ``testing/env-conformance.ts``).

Runner-independent cases for any ``ExecutionEnv``. `with_env` must call and
await its callback exactly once per case with an environment whose ``cwd`` is a
fresh, empty, writable directory.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from ..env import (
    BACKGROUND_CONTEXT,
    Context,
    ExecutionEnv,
    FileError,
    FileInfo,
    Result,
    ShellExecOptions,
    ShellOutputInfo,
    ShellOutputWindow,
    WatchChange,
    WatchTarget,
    aborted_context,
    get_or_throw,
    with_abort_signal,
)
from .types import ConformanceAssertions, EnvConformanceCase, EnvConformanceOptions

context = BACKGROUND_CONTEXT

_A = ConformanceAssertions()
assert_obj = _A

# The active run's shell (program and flags), set by create_env_conformance.
_SHELL = ["sh", "-c"]


def _decoder(data: bytes) -> str:
    return data.decode("utf-8")


def _encode(text: str) -> bytes:
    return text.encode("utf-8")


async def _error_code(operation) -> Optional[str]:
    """The stable code of a fallible operation's ``err`` result; ``None`` when it succeeded."""
    result = await operation
    if result.ok:
        return None
    return getattr(result.error, "code", None)


async def _read_all(env: ExecutionEnv, path: str, max_entries: int) -> Dict[str, Any]:
    reader = get_or_throw(await env.open_dir_reader(path, context))
    pages: List[List[FileInfo]] = []
    try:
        for _page in range(1000):
            next_page = get_or_throw(await reader.next(max_entries, context))
            pages.append(next_page["entries"])
            if next_page["done"]:
                return {"pages": pages, "done": True}
        return {"pages": pages, "done": False}
    finally:
        await reader.close(context)


def _covers(change: WatchChange, path: str) -> bool:
    """Whether a change reports ``path``: an overflow, or a reported path at or above it."""
    if "overflow" in change:
        return True
    if "paths" not in change:
        return False
    return any(
        path == reported or path.startswith(f"{reported}/") or path.startswith(f"{reported}\\")
        for reported in change["paths"]
    )


async def _watching(env: ExecutionEnv, targets: Sequence[WatchTarget], run) -> None:
    """Watch ``targets`` while ``run`` changes files, with a helper that waits for a change."""
    changes: List[WatchChange] = []
    watcher = get_or_throw(await env.watch(targets, changes.append, context))
    try:

        async def absolute(path: str) -> str:
            return get_or_throw(await env.absolute_path(path, context))

        async def expect_change(path: str, change) -> None:
            target = await absolute(path)
            start = len(changes)
            await change()
            deadline = time.monotonic() + 3.0
            while not any(_covers(entry, target) for entry in changes[start:]):
                if time.monotonic() > deadline:
                    observed = json.dumps(changes[start:], default=repr)
                    raise AssertionError(f"No change reported {target}; got {observed}")
                await asyncio.sleep(0.02)

        await run({"changes": changes, "absolute": absolute, "expect_change": expect_change})
    finally:
        await watcher.close(context)


async def exec_collect(env: ExecutionEnv, command, cwd: Optional[str] = None) -> Dict[str, Any]:
    output: Dict[str, str] = {"stdout": "", "stderr": ""}

    def on_output(text: str, _context: Context, info: ShellOutputInfo) -> None:
        output[info.stream] += text

    exec_options = ShellExecOptions(cwd=cwd, on_output=on_output)
    result = await env.exec(command, exec_options, context)
    return {"result": result, "stdout": output["stdout"], "stderr": output["stderr"]}


def create_env_conformance(options: EnvConformanceOptions) -> List[EnvConformanceCase]:
    shell = options.shell or ["sh", "-c"]
    symlinks = True if options.symlinks is None else options.symlinks
    global _SHELL
    _SHELL = list(shell)

    def case(name: str, test, timeout_seconds: Optional[float] = None) -> EnvConformanceCase:
        return EnvConformanceCase(
            name=name,
            run=lambda: options.with_env(test),
            timeout_seconds=timeout_seconds,
        )

    # Watch cases wait up to three seconds per step, longer than test runners allow by default.
    def watch_case(name: str, test) -> EnvConformanceCase:
        return case(name, test, timeout_seconds=30.0)

    cases: List[EnvConformanceCase] = [
        case("binary reader reads byte ranges of the opened file", _case_binary_read),
        case("binary reader scans lines like decoding the whole file", _case_scan_lines),
        case("binary reader keeps reading the file it opened after a rename", _case_rename_read),
        case("binary reader refuses directories, missing files and aborted opens", _case_reader_refusals),
        case("directory reader pages every entry exactly once", _case_dir_pages),
        case("directory reader reports the end and refuses use after close", _case_dir_close),
        case("directory reader refuses missing paths and files", _case_dir_refusals),
        case("directory reader skips entries removed during enumeration", _case_dir_skip_removed),
        watch_case(
            "watch reports a missing file's creation, changes, replacement and removal",
            _case_watch_file,
        ),
        watch_case("watch reports a missing target whose ancestors are created", _case_watch_ancestors),
        watch_case("watch follows directories created together with their contents", _case_watch_new_dirs),
        watch_case(
            "watch keeps watching a path whose parent is renamed and recreated",
            _case_watch_parent_renamed,
        ),
        watch_case(
            "watch skips excluded entries and reports a rename out of them",
            _case_watch_excludes,
        ),
        watch_case("watch stops reporting once closed", _case_watch_closed),
        case("argv exec passes arguments to the program without shell parsing", _case_argv_exec),
        case("exec reports the stream of every chunk in both forms", _case_exec_streams),
        case("argv exec honors cwd and exit codes", _case_exec_cwd),
        case("argv exec reports missing programs and empty argv as spawn errors", _case_exec_spawn_errors),
        case("windowed exec keeps the exact tail and counts what it skips", _case_exec_window),
        case("argv exec distinguishes timeout from abort", _case_exec_timeout_abort),
    ]
    if symlinks:
        cases.append(
            case(
                "binary reader follows symlinks unless noFollow refuses the final one",
                _case_symlinks,
            )
        )
    return cases


async def _case_binary_read(env: ExecutionEnv) -> None:
    get_or_throw(await env.write_file("data.txt", "hello world", context))
    reader = get_or_throw(await env.open_binary_reader("data.txt", None, context))
    info = get_or_throw(await reader.info(context))
    assert_obj.partial_deep_equal(info, {"name": "data.txt", "kind": "file", "size": 11})
    assert_obj.strict_equal(_decoder(get_or_throw(await reader.read(0, 5, context))), "hello")
    assert_obj.strict_equal(_decoder(get_or_throw(await reader.read(6, 100, context))), "world")
    assert_obj.strict_equal(len(get_or_throw(await reader.read(11, 4, context))), 0)
    assert_obj.strict_equal(len(get_or_throw(await reader.read(50, 1, context))), 0)
    assert_obj.strict_equal(len(get_or_throw(await reader.read(3, 0, context))), 0)
    assert_obj.strict_equal(await _error_code(reader.read(-1, 1, context)), "invalid")
    assert_obj.strict_equal(await _error_code(reader.read(0, 1.5, context)), "invalid")
    assert_obj.strict_equal(await _error_code(reader.read(0, 1, aborted_context())), "aborted")
    await reader.close(context)
    await reader.close(context)
    assert_obj.strict_equal(await _error_code(reader.read(0, 1, context)), "invalid")
    assert_obj.strict_equal(await _error_code(reader.info(context)), "invalid")


async def _case_scan_lines(env: ExecutionEnv) -> None:
    # A byte-order mark, an invalid sequence before a newline, an empty line, a later U+FEFF, and no final newline.
    data = bytes(
        [0xEF, 0xBB, 0xBF, 0x61, 0x0A, 0xE2, 0x82, 0x0A, 0x0A, 0xEF, 0xBB, 0xBF, 0x62, 0x0A, 0xC3, 0xA9]
    )
    get_or_throw(await env.write_file("lines.txt", data, context))
    lines = data.decode("utf-8", "replace")
    if lines.startswith("\ufeff"):
        lines = lines[1:]  # TextDecoder drops a leading byte-order mark.
    lines = lines.split("\n")
    reader = get_or_throw(await env.open_binary_reader("lines.txt", None, context))
    try:
        ranges = [(0, None), (0, 1), (1, 3), (2, 3), (3, None), (4, 9)]
        for start_line, end_line in ranges:
            options: Dict[str, int] = {"start_line": start_line}
            if end_line is not None:
                options["end_line"] = end_line
            scan = get_or_throw(await reader.scan_lines(options, context))
            selected = lines[start_line:end_line]

            def range_text(start: int, end: int) -> str:
                text = data[start:end]
                if start == 0 and text.startswith(b"\xef\xbb\xbf"):
                    text = text[3:]
                return text.decode("utf-8", "replace")

            assert_obj.strict_equal(scan.newlines, len(lines) - 1)
            assert_obj.strict_equal(range_text(scan.start, scan.end), "\n".join(selected))
            assert_obj.strict_equal(scan.selected_bytes, len(_encode("\n".join(selected))))
            assert_obj.strict_equal(range_text(scan.start, scan.first_line_end), lines[start_line])
            assert_obj.strict_equal(scan.first_line_bytes, len(_encode(lines[start_line])))
        scan = get_or_throw(await reader.scan_lines({"start_line": 9}, context))
        assert_obj.partial_deep_equal(scan, {"start": len(data), "end": len(data), "selected_bytes": 0})
        assert_obj.strict_equal(
            await _error_code(reader.scan_lines({"start_line": 2, "end_line": 2}, context)), "invalid"
        )
    finally:
        await reader.close(context)


async def _case_rename_read(env: ExecutionEnv) -> None:
    get_or_throw(await env.write_file("a.txt", "one", context))
    reader = get_or_throw(await env.open_binary_reader("a.txt", None, context))
    try:
        get_or_throw(await env.rename_file("a.txt", "b.txt", context))
        get_or_throw(await env.write_file("a.txt", "two", context))
        assert_obj.strict_equal(_decoder(get_or_throw(await reader.read(0, 10, context))), "one")
    finally:
        await reader.close(context)


async def _case_reader_refusals(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("dir", None, context))
    get_or_throw(await env.write_file("file.txt", "x", context))
    assert_obj.strict_equal(await _error_code(env.open_binary_reader("dir", None, context)), "is_directory")
    assert_obj.strict_equal(await _error_code(env.open_binary_reader("missing.txt", None, context)), "not_found")
    assert_obj.strict_equal(
        await _error_code(env.open_binary_reader("file.txt", None, aborted_context())), "aborted"
    )


async def _case_dir_pages(env: ExecutionEnv) -> None:
    names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
    for name in names:
        get_or_throw(await env.write_file(name, name, context))
    get_or_throw(await env.create_dir("sub", None, context))
    listing = await _read_all(env, ".", 2)
    assert_obj.ok(listing["done"], "directory reader reached the end")
    for page in listing["pages"]:
        assert_obj.ok(len(page) <= 2, "page within maxEntries")
    entries = [entry for page in listing["pages"] for entry in page]
    assert_obj.deep_equal(sorted(entry.name for entry in entries), sorted([*names, "sub"]))
    sub = next(entry for entry in entries if entry.name == "sub")
    assert_obj.strict_equal(sub.kind, "directory")
    a_entry = next(entry for entry in entries if entry.name == "a.txt")
    assert_obj.partial_deep_equal(a_entry, {"kind": "file", "size": 5})


async def _case_dir_close(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("empty", None, context))
    reader = get_or_throw(await env.open_dir_reader("empty", context))
    assert_obj.deep_equal(get_or_throw(await reader.next(10, context)), {"entries": [], "done": True})
    assert_obj.deep_equal(get_or_throw(await reader.next(10, context)), {"entries": [], "done": True})
    assert_obj.strict_equal(await _error_code(reader.next(0, context)), "invalid")
    assert_obj.strict_equal(await _error_code(reader.next(1, aborted_context())), "aborted")
    await reader.close(context)
    await reader.close(context)
    assert_obj.strict_equal(await _error_code(reader.next(1, context)), "invalid")


async def _case_dir_refusals(env: ExecutionEnv) -> None:
    get_or_throw(await env.write_file("file.txt", "x", context))
    assert_obj.strict_equal(await _error_code(env.open_dir_reader("missing", context)), "not_found")
    assert_obj.strict_equal(await _error_code(env.open_dir_reader("file.txt", context)), "not_directory")
    assert_obj.strict_equal(await _error_code(env.open_dir_reader(".", aborted_context())), "aborted")


async def _case_dir_skip_removed(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("dir", None, context))
    for name in ["x", "y", "z"]:
        get_or_throw(await env.write_file(f"dir/{name}", name, context))
    reader = get_or_throw(await env.open_dir_reader("dir", context))
    try:
        for name in ["x", "y", "z"]:
            get_or_throw(await env.remove(f"dir/{name}", None, context))
        entries: List[FileInfo] = []
        for _page in range(10):
            next_page = get_or_throw(await reader.next(10, context))
            entries.extend(next_page["entries"])
            if next_page["done"]:
                break
        assert_obj.deep_equal(entries, [])
    finally:
        await reader.close(context)


async def _case_watch_file(env: ExecutionEnv) -> None:
    async def run(helpers) -> None:
        expect_change = helpers["expect_change"]
        await expect_change("AGENTS.md", lambda: env.write_file("AGENTS.md", "one", context))
        await expect_change("AGENTS.md", lambda: env.write_file("AGENTS.md", "two!", context))

        async def replace() -> None:
            get_or_throw(await env.write_file("AGENTS.md.tmp", "three", context))
            get_or_throw(await env.rename_file("AGENTS.md.tmp", "AGENTS.md", context))

        # Editors replace a file by renaming a new one over it.
        await expect_change("AGENTS.md", replace)
        await expect_change("AGENTS.md", lambda: env.write_file("AGENTS.md", "four", context))

        async def remove_file() -> None:
            get_or_throw(await env.remove("AGENTS.md", None, context))

        await expect_change("AGENTS.md", remove_file)

    await _watching(env, [WatchTarget(path="AGENTS.md")], run)


async def _case_watch_ancestors(env: ExecutionEnv) -> None:
    async def run(helpers) -> None:
        await helpers["expect_change"](
            "a/b/c/AGENTS.md", lambda: env.write_file("a/b/c/AGENTS.md", "x", context)
        )

    await _watching(env, [WatchTarget(path="a/b/c/AGENTS.md")], run)


async def _case_watch_new_dirs(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("skills", None, context))

    async def run(helpers) -> None:
        expect_change = helpers["expect_change"]
        # Written before any watcher on the new directories can exist.
        await expect_change("skills/a/b/SKILL.md", lambda: env.write_file("skills/a/b/SKILL.md", "one", context))
        await expect_change("skills/a/b/SKILL.md", lambda: env.write_file("skills/a/b/SKILL.md", "two!", context))
        await expect_change(
            "skills/a/b/c/SKILL.md", lambda: env.write_file("skills/a/b/c/SKILL.md", "deeper", context)
        )

    await _watching(env, [WatchTarget(path="skills", recursive=True)], run)


async def _case_watch_parent_renamed(env: ExecutionEnv) -> None:
    get_or_throw(await env.write_file("proj/.pi/skills/x.md", "x", context))

    async def run(helpers) -> None:
        expect_change = helpers["expect_change"]
        await expect_change("proj/.pi/skills", lambda: env.rename_file("proj/.pi", "proj/old", context))
        await expect_change("proj/.pi/skills/y.md", lambda: env.write_file("proj/.pi/skills/y.md", "y", context))
        await expect_change("proj/.pi/skills/y.md", lambda: env.write_file("proj/.pi/skills/y.md", "yy", context))

    await _watching(env, [WatchTarget(path="proj/.pi/skills", recursive=True)], run)


async def _case_watch_excludes(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("skills", None, context))
    targets = [WatchTarget(path="skills", recursive=True, exclude={"hidden": True, "names": ["node_modules"]})]

    async def run(helpers) -> None:
        changes = helpers["changes"]
        expect_change = helpers["expect_change"]
        absolute = helpers["absolute"]
        get_or_throw(await env.write_file("skills/node_modules/dep/SKILL.md", "dep", context))
        get_or_throw(await env.write_file("skills/.SKILL.md.tmp", "draft", context))

        async def rename_out() -> None:
            get_or_throw(await env.rename_file("skills/.SKILL.md.tmp", "skills/SKILL.md", context))

        await expect_change("skills/SKILL.md", rename_out)
        hidden = [
            await absolute("skills/node_modules"),
            await absolute("skills/.SKILL.md.tmp"),
        ]
        for change in changes:
            if "paths" not in change:
                continue
            for path in change["paths"]:
                assert_obj.ok(
                    not any(path == excluded or path.startswith(excluded) for excluded in hidden),
                    f"excluded {path}",
                )

    await _watching(env, targets, run)


async def _case_watch_closed(env: ExecutionEnv) -> None:
    changes: List[WatchChange] = []
    watcher = get_or_throw(await env.watch([WatchTarget(path="file.txt")], changes.append, context))
    assert_obj.ok(watcher.mode in ("native", "polling"), "watcher reports its mode")
    await watcher.close(context)
    await watcher.close(context)
    get_or_throw(await env.write_file("file.txt", "x", context))
    await asyncio.sleep(0.3)
    assert_obj.deep_equal(changes, [])


async def _case_argv_exec(env: ExecutionEnv) -> None:
    hostile = "it's $(touch pwned) `touch pwned` *; touch pwned"
    collected = await exec_collect(env, [*_SHELL, 'printf "%s|%s" "$1" "$2"', "argv0", hostile, "a b"])
    assert_obj.strict_equal(get_or_throw(collected["result"]).exit_code, 0)
    assert_obj.strict_equal(collected["stdout"], f"{hostile}|a b")
    assert_obj.strict_equal(get_or_throw(await env.exists("pwned", context)), False)


async def _case_exec_streams(env: ExecutionEnv) -> None:
    script = "printf out; printf err >&2; printf more"
    argv = await exec_collect(env, [*_SHELL, script])
    assert_obj.strict_equal(get_or_throw(argv["result"]).exit_code, 0)
    assert_obj.strict_equal(argv["stdout"], "outmore")
    assert_obj.strict_equal(argv["stderr"], "err")
    string = await exec_collect(env, script)
    assert_obj.strict_equal(get_or_throw(string["result"]).exit_code, 0)
    assert_obj.strict_equal(string["stdout"], "outmore")
    assert_obj.strict_equal(string["stderr"], "err")


async def _case_exec_cwd(env: ExecutionEnv) -> None:
    get_or_throw(await env.create_dir("sub", None, context))
    made = await exec_collect(env, [*_SHELL, "printf x > made.txt; exit 3"], "sub")
    assert_obj.strict_equal(get_or_throw(made["result"]).exit_code, 3)
    assert_obj.strict_equal(get_or_throw(await env.read_text_file("sub/made.txt", context)), "x")


async def _case_exec_spawn_errors(env: ExecutionEnv) -> None:
    assert_obj.strict_equal(
        await _error_code(env.exec(["pi-durable-conformance-missing-program"], None, context)), "spawn_error"
    )
    assert_obj.strict_equal(await _error_code(env.exec([], None, context)), "spawn_error")


async def _case_exec_window(env: ExecutionEnv) -> None:
    lines = 2000
    window = {"max_bytes": 200, "max_lines": 5, "min_interval_ms": 0, "bytes_per_second": 1_000_000_000}
    counts = {"bytes": 0, "newlines": 0, "tail": ""}

    def on_output(text: str, _context: Context, info: ShellOutputInfo) -> None:
        if info.skipped is not None:
            counts["bytes"] += info.skipped.bytes
            counts["newlines"] += info.skipped.newlines
            after = len(_encode(text))
            after_newlines = text.count("\n")
            assert_obj.ok(
                after > window["max_bytes"] or after_newlines > window["max_lines"],
                "a skip is followed by more than the window",
            )
            counts["tail"] = ""
        counts["bytes"] += len(_encode(text))
        counts["newlines"] += text.count("\n")
        counts["tail"] += text

    options = ShellExecOptions(window=ShellOutputWindow(**window), on_output=on_output)
    script = f"i=0; while [ $i -lt {lines} ]; do echo line-$i; i=$((i+1)); done"
    result = await env.exec([*_SHELL, script], options, context)
    assert_obj.strict_equal(get_or_throw(result).exit_code, 0)
    expected = "".join(f"line-{index}\n" for index in range(lines))
    assert_obj.strict_equal(counts["bytes"], len(expected))
    assert_obj.strict_equal(counts["newlines"], lines)
    last_lines = "".join(f"line-{index}\n" for index in range(lines - window["max_lines"], lines))
    assert_obj.ok(counts["tail"].endswith(last_lines), "the delivered output ends with the tail")


async def _case_exec_timeout_abort(env: ExecutionEnv) -> None:
    assert_obj.strict_equal(
        await _error_code(env.exec([*_SHELL, "sleep 2"], ShellExecOptions(timeout=0.1), context)), "timeout"
    )
    controller = asyncio.Event()
    running = env.exec([*_SHELL, "sleep 2"], None, with_abort_signal(controller, context))
    asyncio.get_running_loop().call_later(0.1, controller.set)
    assert_obj.strict_equal(await _error_code(running), "aborted")


async def _case_symlinks(env: ExecutionEnv) -> None:
    get_or_throw(await env.write_file("target.txt", "target", context))
    get_or_throw(await env.create_dir("sub", None, context))
    get_or_throw(await env.write_file("sub/inner.txt", "inner", context))
    linked = await exec_collect(env, [*_SHELL, "ln -s target.txt link.txt && ln -s sub dirlink"])
    assert_obj.strict_equal(get_or_throw(linked["result"]).exit_code, 0)

    followed = get_or_throw(await env.open_binary_reader("link.txt", None, context))
    assert_obj.strict_equal(_decoder(get_or_throw(await followed.read(0, 10, context))), "target")
    await followed.close(context)

    assert_obj.strict_equal(
        await _error_code(env.open_binary_reader("link.txt", {"no_follow": True}, context)), "invalid"
    )

    # Only the final component is refused; earlier symlinked directories still resolve.
    inner = get_or_throw(await env.open_binary_reader("dirlink/inner.txt", {"no_follow": True}, context))
    assert_obj.strict_equal(_decoder(get_or_throw(await inner.read(0, 10, context))), "inner")
    await inner.close(context)
