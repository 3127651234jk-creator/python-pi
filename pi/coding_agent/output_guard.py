"""Stdout takeover for protocol modes (port of output-guard.ts).

JSON and RPC modes carry a line-based protocol on stdout. ``take_over_stdout``
rebinds ``sys.stdout`` to stderr so stray ``print`` output (extensions, libs)
cannot corrupt the protocol stream; protocol writers use ``raw_stdout()``
instead. ``restore_stdout`` puts the original stream back.
"""

import sys
from typing import Optional


class _BinaryForwarder:
    """Binary view of stderr, so ``sys.stdout.buffer`` cannot reach real stdout."""

    def write(self, data):
        stream = sys.stderr
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            return stream.write(data.decode("utf-8", "replace"))
        return buffer.write(data)

    def flush(self):
        stream = sys.stderr
        buffer = getattr(stream, "buffer", None)
        return (buffer if buffer is not None else stream).flush()

    def __getattr__(self, name):
        stream = sys.stderr
        return getattr(getattr(stream, "buffer", None) or stream, name)


class _Forwarder:
    """Minimal file object forwarding writes to stderr."""

    def __init__(self, original) -> None:
        self.original = original

    @property
    def buffer(self):
        # Delegating ``buffer`` to the original stream would hand out the real
        # stdout buffer; binary writes would then bypass the protocol guard.
        return _BinaryForwarder()

    def write(self, text):
        return sys.stderr.write(text)

    def flush(self):
        return sys.stderr.flush()

    def isatty(self):
        return False

    def fileno(self):
        # The guard is stderr, so fd-level writes (os.write, logging handlers)
        # must target stderr's descriptor rather than stdout's.
        return sys.stderr.fileno()

    def __getattr__(self, name):
        return getattr(self.original, name)


_state: Optional[object] = None


def take_over_stdout() -> None:
    global _state
    if _state is not None:
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _state = sys.stdout
    sys.stdout = _Forwarder(_state)


def restore_stdout() -> None:
    global _state
    if _state is None:
        return
    if sys.stdout is not _state and not isinstance(sys.stdout, _Forwarder):
        # Someone else replaced stdout (e.g. the fullscreen TUI); leave theirs.
        _state = None
        return
    sys.stdout = _state
    _state = None


def is_stdout_taken_over() -> bool:
    return _state is not None


def raw_stdout():
    """The real stdout stream, even while taken over."""
    return _state if _state is not None else sys.stdout


def write_raw_stdout(text: str) -> None:
    """Write protocol output to the real stdout regardless of takeover."""
    if not text:
        return
    stream = raw_stdout()
    stream.write(text)
    stream.flush()
