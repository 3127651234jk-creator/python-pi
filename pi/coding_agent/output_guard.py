"""Stdout takeover for protocol modes (port of output-guard.ts).

JSON and RPC modes carry a line-based protocol on stdout. ``take_over_stdout``
rebinds ``sys.stdout`` to stderr so stray ``print`` output (extensions, libs)
cannot corrupt the protocol stream; protocol writers use ``raw_stdout()``
instead. ``restore_stdout`` puts the original stream back.
"""

import sys
from typing import Optional


class _Forwarder:
    """Minimal file object forwarding writes to stderr."""

    def __init__(self, original) -> None:
        self.original = original

    def write(self, text):
        return sys.stderr.write(text)

    def flush(self):
        return sys.stderr.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.original, name)


_state: Optional[object] = None


def take_over_stdout() -> None:
    global _state
    if _state is not None:
        return
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
