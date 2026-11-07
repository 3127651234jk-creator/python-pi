"""Chord delta apply semantics, moved to ``pi.delta`` (this module re-exports).

Storage replays ``delta`` document revisions over their bases at materialization
time; the generic apply, path validation, and wire codec live in ``pi.delta``
alongside the chord service layer that shares them.
"""

from __future__ import annotations

from ..delta import (
    RESERVED_SEGMENTS,
    PathError,
    RevisionValidator,
    UnsafePathError,
    WireDecoder,
    WireEncoder,
    apply_ops,
    assert_safe_path,
    assert_valid_op,
    assert_valid_wire_op,
    is_base,
)

__all__ = [
    "RESERVED_SEGMENTS",
    "PathError",
    "RevisionValidator",
    "UnsafePathError",
    "WireDecoder",
    "WireEncoder",
    "apply_ops",
    "assert_safe_path",
    "assert_valid_op",
    "assert_valid_wire_op",
    "is_base",
]
