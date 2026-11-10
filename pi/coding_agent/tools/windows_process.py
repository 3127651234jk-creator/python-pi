"""Windows job ownership for shell process trees."""
from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import json
import sys
from typing import List, Tuple


class WindowsJob:
    def __init__(self) -> None:
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(self._api, name)
            function.argtypes = args
            function.restype = result
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid: int) -> None:
        process = self._api.OpenProcess(0x0100 | 0x0001, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self._api.AssignProcessToJobObject(self._handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._api.CloseHandle(process)

    def terminate(self) -> None:
        if not self._api.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle:
            self._api.CloseHandle(self._handle)
            self._handle = None


# The helper cannot launch the command until it is assigned to the job.
# Descendants then inherit job membership even if their parents exit.
_BOOTSTRAP = """
import json
import subprocess
import sys

argv = json.loads(sys.stdin.buffer.readline())
try:
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
except OSError as error:
    print(json.dumps([error.errno, error.strerror, error.filename]), file=sys.stderr, flush=True)
    sys.exit(1)
print("null", file=sys.stderr, flush=True)
sys.exit(proc.wait())
"""


async def spawn_windows_command(argv: List[str], cwd: str) -> Tuple[asyncio.subprocess.Process, WindowsJob]:
    job = WindowsJob()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _BOOTSTRAP,
            cwd=cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        job.assign(proc.pid)
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(argv) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        assert proc.stderr is not None
        status = await proc.stderr.readline()
        if not status:
            raise RuntimeError("Windows command helper exited before starting the command")
        error = json.loads(status)
        if error is not None:
            raise OSError(*error)
        return proc, job
    except BaseException:
        try:
            job.terminate()
            if proc is not None:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
        finally:
            job.close()
        raise
