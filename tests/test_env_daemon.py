"""Connection, RemoteExecutionEnv, and RemoteWatcher behaviors beyond the conformance suite:
reconnect after loss, handle-session pinning, exec abort/timeout/cleanup, spill, and watcher
reconnection (``packages/env``'s ``test/differential`` and unit concerns, in Python form).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.env import Context, FileError, ShellExecOptions, ShellSpillOptions, WatchTarget, get_or_throw
from pi.env import Connection, RemoteExecutionEnv, RemoteWatchOptions, is_connection_lost, to_file_error
from pi.env.connection import RemoteError
from pi.env.remote_env import RemotePaths, _file_url_to_path


def daemon_command():
    return [sys.executable, "-m", "pi.env.daemon"]


def _write_fake_daemon(path, sync_delay=0.0, stderr_bytes=0):
    """A minimal protocol speaker: optional startup delay and stderr flood, then hello replies."""
    path.write_text(
        "\n".join(
            [
                "import json, struct, sys, time",
                "token = sys.argv[sys.argv.index('--token') + 1]",
                "time.sleep({delay})".format(delay=sync_delay),
                "sys.stdout.buffer.write(f'PI-ENV {token}\\n'.encode('ascii'))",
                "sys.stdout.buffer.flush()",
                "flood = {flood}".format(flood=stderr_bytes),
                "block = b'x' * 65536",
                "while flood > 0:",
                "    sys.stderr.buffer.write(block[:min(flood, len(block))])",
                "    flood -= len(block)",
                "sys.stderr.buffer.flush()",
                "while True:",
                "    header = sys.stdin.buffer.read(4)",
                "    if len(header) < 4:",
                "        break",
                "    length = int.from_bytes(header, 'big')",
                "    body = sys.stdin.buffer.read(length)",
                "    request_id = int.from_bytes(body[1:5], 'big')",
                "    json_length = int.from_bytes(body[5:9], 'big')",
                "    request = json.loads(body[9:9 + json_length])",
                "    if request.get('op') == 'hello':",
                "        payload = json.dumps({'protocol': 1, 'version': 'test', 'os': 'linux', 'arch': 'x86_64',",
                "                               'home': '/', 'tmpdir': '/tmp', 'separator': '/', 'cwd': '/',",
                "                               'driveCwds': {}, 'pid': 1}).encode()",
                "        sys.stdout.buffer.write(struct.pack('>IBII', 9 + len(payload), 2, request_id, len(payload)) + payload)",
                "        sys.stdout.buffer.flush()",
            ]
        ),
        encoding="utf-8",
    )


def make_env(cwd):
    connection = Connection(daemon_command())
    environment = RemoteExecutionEnv(connection=connection, id="test-env", cwd=str(cwd))
    return connection, environment


async def _kill_daemon(connection: Connection) -> None:
    """Simulate a lost connection: the daemon dies without saying goodbye."""
    session = connection._session
    assert session is not None
    session.process.kill()
    await asyncio.wait_for(session.process.wait(), timeout=10)
    await asyncio.sleep(0.1)


def test_paths():
    windows = RemotePaths(True)
    assert windows.resolve_relative("C:\\work", "sub\\file") == "C:\\work\\sub\\file"
    assert windows.resolve_absolute("C:\\work\\..\\other") == "C:\\other"
    assert windows.join("C:\\work", "a") == "C:\\work\\a"
    posix = RemotePaths(False)
    assert posix.resolve_relative("/work", "sub/file") == "/work/sub/file"
    assert posix.resolve_absolute("/work/../other") == "/other"
    assert _file_url_to_path("file:///C:/x/y", True) == "C:\\x\\y"
    assert _file_url_to_path("file://server/x", True) == "\\\\server\\x"
    assert _file_url_to_path("file:///etc/hosts", False) == "/etc/hosts"
    assert _file_url_to_path("file://server/etc/hosts", False) == "//server/etc/hosts"
    assert _file_url_to_path("not-a-url", True) is None


def test_to_file_error():
    from pi.env.connection import RemoteError

    assert to_file_error(FileError("aborted", "aborted")).code == "aborted"
    assert to_file_error(RemoteError({"code": "ENOENT", "message": "m"})).code == "not_found"
    assert to_file_error(RemoteError({"code": "EACCES", "message": "m"})).code == "permission_denied"
    assert to_file_error(RemoteError({"code": "ENOTDIR", "message": "m"})).code == "not_directory"
    assert to_file_error(RemoteError({"code": "EISDIR", "message": "m"})).code == "is_directory"
    assert to_file_error(RemoteError({"code": "SYMLINK", "message": "m"})).code == "invalid"
    assert to_file_error(RemoteError({"code": "NOT_REGULAR", "message": "m"})).code == "invalid"
    assert to_file_error(RemoteError({"code": "EWHATEVER", "message": "m"})).code == "unknown"
    assert to_file_error(RuntimeError("boom")).code == "unknown"
    error = to_file_error(RemoteError({"code": "ENOENT", "message": "m", "path": "/a"}))
    assert error.path == "/a"


def test_reconnect_after_loss(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            info = await connection.info()
            assert info.protocol == 1
            first_session = connection._session.id
            await _kill_daemon(connection)
            # A fresh request starts a new daemon.
            info = await connection.info()
            assert connection._session.id == first_session + 1
            get_or_throw(await environment.write_file("a.txt", "hello", Context()))
        finally:
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_lost_requests_fail_and_next_request_restarts(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            get_or_throw(await environment.write_file("a.txt", "abc", Context()))
            reader = get_or_throw(await environment.open_binary_reader("a.txt", None, Context()))
            session_id = await connection.session_id()
            await _kill_daemon(connection)
            # Requests on handles of the lost session fail instead of reaching a daemon that never
            # opened the handle.
            result = await reader.read(0, 3, Context())
            assert not result.ok
            assert result.error.code == "unknown"
            assert is_connection_lost(result.error.__cause__)
            assert connection._session is None  # no daemon was started for the failed read
            # Ordinary requests start a new daemon; the old handle still refuses.
            get_or_throw(await environment.write_file("b.txt", "abc", Context()))
            result = await reader.read(0, 3, Context())
            assert not result.ok
            assert result.error.code == "unknown"
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_exec_abort(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            abort_signal = asyncio.Event()
            context = Context(abort_signal=abort_signal)
            command = [sys.executable, "-c", "import time; time.sleep(30)"]
            task = asyncio.ensure_future(environment.exec(command, None, context))
            await asyncio.sleep(0.5)
            abort_signal.set()
            result = await task
            assert not result.ok
            assert result.error.code == "aborted"
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_exec_timeout(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            command = [sys.executable, "-c", "import time; time.sleep(30)"]
            result = await environment.exec(command, ShellExecOptions(timeout=0.5), Context())
            assert not result.ok
            assert result.error.code == "timeout"
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_exec_cleanup_kills_without_aborting(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            command = [sys.executable, "-c", "import time; time.sleep(30)"]
            task = asyncio.ensure_future(environment.exec(command, None, Context()))
            await asyncio.sleep(0.5)
            await environment.cleanup(Context())
            result = await task
            # Killed without aborting: the command settles with the killed process's status.
            if result.ok:
                assert result.value.exit_code in (1, 137)
            else:
                assert result.error.code != "aborted"
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_exec_spill(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            payload = "x" * 200_000
            get_or_throw(await environment.write_file("big.txt", payload, Context()))
            result = await environment.exec(
                [sys.executable, "-c", "import sys; sys.stdout.write(open(sys.argv[1]).read())", str(tmp_path / "big.txt")],
                ShellExecOptions(spill=ShellSpillOptions(after_bytes=1024, after_lines=10)),
                Context(),
            )
            get_or_throw(result)
            spill_path = result.value.spill_path
            assert spill_path, "expected the output to spill"
            contents = get_or_throw(await environment.read_text_file(spill_path, Context()))
            assert contents == payload
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_exists(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            get_or_throw(await environment.write_file("a.txt", "abc", Context()))
            assert get_or_throw(await environment.exists("a.txt", Context())) is True
            assert get_or_throw(await environment.exists("missing.txt", Context())) is False
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_watcher_reports_overflow_after_reconnect(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            changes = []
            watcher = get_or_throw(
                await environment.watch([WatchTarget(path=".")], changes.append, Context())
            )
            await asyncio.sleep(0.3)  # let the first connection settle
            await _kill_daemon(connection)
            (tmp_path / "late.txt").write_text("hi")
            # The watcher reconnects (its own daemon) and reports that changes were missed.
            deadline = asyncio.get_running_loop().time() + 10
            while asyncio.get_running_loop().time() < deadline:
                covered = any(
                    "overflow" in change
                    or any(str(tmp_path / "late.txt").startswith(reported) for reported in change.get("paths", []))
                    for change in changes
                )
                if covered:
                    break
                await asyncio.sleep(0.1)
            assert covered, f"no change covered late.txt; got {changes!r}"
            await watcher.close(Context())
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_stderr_is_drained_without_on_log(tmp_path):
    """A daemon that floods stderr must not block: the client always reads the pipe."""

    async def main():
        fake = tmp_path / "fake_daemon.py"
        _write_fake_daemon(fake, stderr_bytes=4 * 1024 * 1024)
        connection = Connection([sys.executable, str(fake)])
        try:
            info = await asyncio.wait_for(connection.info(), timeout=10)
            assert info.protocol == 1
        finally:
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_cancelling_a_wait_does_not_cancel_the_shared_start(tmp_path):
    """A cancelled caller must not leave the connection with a cancelled start task."""

    async def main():
        fake = tmp_path / "slow_daemon.py"
        _write_fake_daemon(fake, sync_delay=0.8)
        connection = Connection([sys.executable, str(fake)])
        try:
            waiter = asyncio.ensure_future(connection.info())
            await asyncio.sleep(0.1)
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass
            # The start continued; a later request works.
            info = await asyncio.wait_for(connection.info(), timeout=10)
            assert info.protocol == 1
        finally:
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_decoding_matches_the_local_env(tmp_path):
    """BOM and replacement semantics must agree between the local env and the daemon-backed one."""
    from pi.durable.env import LocalExecutionEnv

    async def main():
        connection, remote = make_env(tmp_path)
        local = LocalExecutionEnv(cwd=str(tmp_path))
        try:
            bom_content = "\ufefffirst\nsecond\n"
            (tmp_path / "bom.txt").write_bytes(bom_content.encode("utf-8"))
            (tmp_path / "invalid.txt").write_bytes(b"bad \xff byte\n")
            for environment in (local, remote):
                # readFile keeps a byte-order mark (StringDecoder semantics).
                assert get_or_throw(await environment.read_text_file("bom.txt", Context())) == bom_content
                # The line reader drops a leading byte-order mark (StreamDecoder semantics).
                assert get_or_throw(await environment.read_text_lines("bom.txt", None, Context())) == [
                    "first",
                    "second",
                ]
                # Invalid sequences decode with U+FFFD, once per maximal subpart.
                assert get_or_throw(await environment.read_text_file("invalid.txt", Context())) == "bad \ufffd byte\n"
            # Command output drops only a leading byte-order mark, per stream.
            outputs = {}

            def on_output(text, _context, info):
                outputs[info.stream] = outputs.get(info.stream, "") + text

            command = [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'\\xef\\xbb\\xbfhi'); sys.stdout.buffer.flush()",
            ]
            for index, environment in enumerate((local, remote)):
                outputs.clear()
                result = get_or_throw(await environment.exec(command, ShellExecOptions(on_output=on_output), Context()))
                assert result.exit_code == 0
                assert outputs.get("stdout") == "hi", (index, outputs)
        finally:
            await remote.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_file_operations_match_the_local_env(tmp_path):
    """Mutations and listings behave like ``LocalExecutionEnv``: same results, same error codes."""
    from pi.durable.env import LocalExecutionEnv

    async def main():
        connection, remote = make_env(tmp_path)
        local = LocalExecutionEnv(cwd=str(tmp_path))
        try:
            for environment in (local, remote):
                get_or_throw(await environment.write_file("f.txt", "one", Context()))
                get_or_throw(await environment.append_file("f.txt", " two", Context()))
                assert get_or_throw(await environment.read_text_file("f.txt", Context())) == "one two"
                get_or_throw(await environment.truncate_file("f.txt", 3, Context()))
                assert get_or_throw(await environment.read_text_file("f.txt", Context())) == "one"
                get_or_throw(await environment.flush_file("f.txt", Context()))
                get_or_throw(await environment.create_dir("a/b", {"recursive": True}, Context()))
                assert get_or_throw(await environment.exists("a/b", Context())) is True
                listing = get_or_throw(await environment.list_dir("a", Context()))
                assert [entry.name for entry in listing] == ["b"]
                canonical = get_or_throw(await environment.canonical_path("a/b", Context()))
                assert canonical.replace("\\", "/").endswith("a/b")
                # A directory without ``recursive`` fails the same way in both environments.
                result = await environment.remove("a", None, Context())
                assert not result.ok and result.error.code == "is_directory"
                get_or_throw(await environment.remove("a", {"recursive": True}, Context()))
                assert get_or_throw(await environment.exists("a", Context())) is False
                # A missing path is fine with ``force`` and refused without it.
                get_or_throw(await environment.remove("missing", {"force": True}, Context()))
                missing = await environment.remove("missing", None, Context())
                assert not missing.ok and missing.error.code == "not_found"
        finally:
            await remote.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_handle_close_waits_for_operations_in_flight():
    """A handle closes after the operations already running on it, like Rust's ``Arc<File>``."""
    from pi.env import daemon as daemon_module

    class _TestHandle(daemon_module.ClosableHandle):
        def __init__(self):
            super().__init__("path")
            self.released = False

        def release(self):
            self.released = True

    handle = _TestHandle()
    with handle.operating():
        handle.close()
        assert handle.released is False  # an operation is still running
    assert handle.released is True

    idle = _TestHandle()
    idle.close()
    assert idle.released is True  # nothing in flight: closes immediately


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects")
def test_windows_job_kills_children_on_close():
    """Children join the daemon's job, so they die with it even without cleanup paths (libuv's model)."""
    from pi.env import daemon as daemon_module

    daemon_module._job[0] = None
    daemon_module._job[1] = False
    job = daemon_module._windows_job()
    assert job, "the job object could not be created"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.2)
        daemon_module.assign_to_job(child.pid)
        time.sleep(0.2)
        assert child.poll() is None
        daemon_module.ctypes.windll.kernel32.CloseHandle(daemon_module.ctypes.c_void_p(job))
        deadline = time.time() + 5
        while child.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        daemon_module._job[0] = None
        daemon_module._job[1] = False


def test_write_with_explicit_null_parents_creates_them(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            target = tmp_path / "nested" / "dir" / "file.txt"
            reply = await connection.request(
                "write", {"path": str(target), "append": False, "parents": None}, payload=b"data"
            )
            assert reply.json == {}
            assert target.read_text() == "data"
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_argv_must_be_strings(tmp_path):
    async def main():
        connection, environment = make_env(tmp_path)
        try:
            try:
                await connection.request("exec", {"argv": [1, 2], "cwd": str(tmp_path)})
            except RemoteError as error:
                assert error.code == "EINVAL", error.code
            else:
                raise AssertionError("expected EINVAL for a non-string argv")
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())


def test_watch_coverage_failure_maps_to_invalid(tmp_path):
    async def main():
        connection = Connection(daemon_command())
        # A coverage budget of zero makes ``LocalFileWatcher.open`` fail before it reports ready.
        environment = RemoteExecutionEnv(
            connection=connection,
            id="test-env",
            cwd=str(tmp_path),
            watch=RemoteWatchOptions(max_directories=0),
        )
        try:
            result = await environment.watch(
                [WatchTarget(path=".")], lambda change: None, Context()
            )
            assert not result.ok
            assert result.error.code == "invalid", result.error
        finally:
            await environment.cleanup(Context())
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())
