"""Chord delta operations over plain JSON (port of ``chord/src/delta`` apply).

Only what storage needs: validating and applying one operation batch to a
mutable value. Paths carry object keys and non-negative integer array indices;
``s``/``d``/``a``/``t`` cannot target the root, ``p`` and ``m`` may. Reserved
segments (``__proto__``-style keys) are ordinary keys in Python dicts, so the
prototype-pollution guard rejects only path shapes, not those names.
"""

from __future__ import annotations

from typing import Any, List, Sequence

RESERVED_SEGMENTS = ("__proto__", "constructor", "prototype")


class PathError(Exception):
    """An operation path does not resolve in the value."""

    def __init__(self, path: Any) -> None:
        super().__init__(f"unresolvable path: {path!r}")
        self.path = path


class UnsafePathError(Exception):
    """A path segment is not a valid key or index."""

    def __init__(self, segment: Any) -> None:
        super().__init__(f"unsafe path segment: {segment!r}")
        self.segment = segment


def assert_valid_op(op: Sequence[Any]) -> None:
    """Verb, arity, and payload shape for one decoded op."""
    if not isinstance(op, (list, tuple)) or len(op) == 0:
        raise TypeError("op is not a tuple")
    verb = op[0]
    if verb == "r":
        if len(op) != 2:
            raise TypeError("r arity")
        return
    if verb in ("s", "d", "a", "t"):
        if verb == "s" and len(op) != 3:
            raise TypeError("s arity")
        if verb == "d" and len(op) != 2:
            raise TypeError("d arity")
        if verb == "a" and (len(op) != 3 or not isinstance(op[2], str)):
            raise TypeError("a shape")
        if verb == "t" and (len(op) != 3 or not _is_index(op[2])):
            raise TypeError("t shape")
        assert_safe_path(op[1], non_empty=True)
        return
    if verb == "p":
        if len(op) != 5 or not _is_index(op[2]) or not _is_index(op[3]) or not isinstance(op[4], list):
            raise TypeError("p shape")
        assert_safe_path(op[1])
        return
    if verb == "m":
        if len(op) != 3:
            raise TypeError("m arity")
        assert_safe_path(op[1])
        _assert_permutation(op[2])
        return
    raise TypeError(f"unknown op verb: {verb!r}")


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def assert_safe_path(path: Any, non_empty: bool = False) -> None:
    if not isinstance(path, (list, tuple)):
        raise TypeError("path is not an array")
    if non_empty and len(path) == 0:
        raise TypeError("path is empty")
    for segment in path:
        if isinstance(segment, str):
            if segment in RESERVED_SEGMENTS:
                raise UnsafePathError(segment)
        elif not _is_index(segment):
            raise UnsafePathError(segment)


def _assert_permutation(value: Any) -> None:
    if not isinstance(value, list) or any(not _is_index(i) or i >= len(value) for i in value):
        raise TypeError("m permutation is not a bijection")
    if len(set(value)) != len(value):
        raise TypeError("m permutation is not a bijection")


def apply_ops(target: Any, ops: Sequence[Sequence[Any]]) -> Any:
    """Apply ops to a plain mutable value; returns the value (``r`` replaces it outright)."""
    root = target
    for op in ops:
        assert_valid_op(op)
        verb = op[0]
        if verb == "r":
            root = op[1]
            continue
        path: List[Any] = list(op[1])
        assert_safe_path(path)

        if verb in ("p", "m"):
            node = root if len(path) == 0 else _resolve(root, path)
            if not isinstance(node, list):
                raise PathError(path)
            if verb == "p":
                index, remove, items = op[2], op[3], op[4]
                if index > len(node):
                    raise UnsafePathError(index)
                del node[index : index + remove]
                for offset in range(0, len(items), 10_000):
                    node[index + offset : index + offset] = items[offset : offset + 10_000]
            else:
                permutation = op[2]
                if len(node) != len(permutation):
                    raise PathError(path)
                previous = list(node)
                for i, source in enumerate(permutation):
                    node[i] = previous[source]
            continue

        # s/d/a/t can never target the root.
        parent = _resolve(root, path[:-1])
        key = path[-1]
        if isinstance(parent, list):
            if not _is_index(key):
                raise UnsafePathError(key)
            if key > len(parent):
                raise UnsafePathError(key)
        elif not isinstance(parent, dict):
            raise PathError(path)

        def read_current() -> Any:
            if isinstance(parent, list):
                return parent[key] if key < len(parent) else None
            return parent.get(key)

        if verb == "s":
            if isinstance(parent, list):
                if key == len(parent):
                    parent.append(op[2])
                else:
                    parent[key] = op[2]
            else:
                parent[key] = op[2]
        elif verb == "d":
            if isinstance(parent, list):
                if key >= len(parent):
                    raise PathError(path)
                del parent[key]
            else:
                if key not in parent:
                    raise PathError(path)
                del parent[key]
        elif verb == "a":
            current = read_current()
            if not isinstance(current, str):
                raise PathError(path)
            _write(parent, key, current + op[2])
        elif verb == "t":
            current = read_current()
            if not isinstance(current, str):
                raise PathError(path)
            # UTF-16 units vs Python code points: equal for the BMP text storage uses.
            _write(parent, key, current[op[2] :])
    return root


def _write(parent: Any, key: Any, value: Any) -> None:
    if isinstance(parent, list):
        parent[key] = value
    else:
        parent[key] = value


def _resolve(root: Any, path: Sequence[Any]) -> Any:
    node = root
    for segment in path:
        if isinstance(node, list):
            if not _is_index(segment):
                raise UnsafePathError(segment)
            if segment >= len(node):
                raise PathError(list(path))
            node = node[segment]
        elif isinstance(node, dict):
            if segment not in node:
                raise PathError(list(path))
            node = node[segment]
        else:
            raise PathError(list(path))
    return node
