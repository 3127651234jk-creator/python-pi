"""Chord delta operations over plain JSON (port of ``chord/src/delta``).

The apply half (``apply_ops``) moved here from ``pi.durable.delta``; this module
adds what a routed boundary needs: the wire codec with path interning
(``["#", id, path]`` emitted on a path's second use), short forms that reuse the
previous op's path within one batch, and a path dictionary that spans batches
until an ``r`` clears it. Also the strict-JSON revision validation used by
service replicas.

Ops are tuples of the form ``[verb, ...]``: ``r`` (replace root), ``s`` (set),
``d`` (delete), ``a`` (string append), ``t`` (string tail-slice from UTF-16
offset), ``p`` (splice), ``m`` (permutation). ``s``/``d``/``a``/``t`` cannot
target the root; ``p`` and ``m`` may. Reserved segments (``__proto__``-style
keys) are ordinary keys in Python dicts, so the prototype-pollution guard
rejects only path shapes, not those names.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Set, Tuple

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


def assert_valid_op(op: Sequence[Any]) -> None:
    """Verb, arity, and payload shape for one decoded op (paths inline, no wire forms)."""
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
    # Silently skipping an unknown verb is how a newer producer's op vanishes.
    raise TypeError(f"unknown op verb: {verb!r}")


def assert_valid_wire_op(op: Sequence[Any]) -> None:
    """The same for the wire grammar: path ids, short forms and ``#`` are legal here."""
    if not isinstance(op, (list, tuple)) or len(op) == 0:
        raise TypeError("op is not a tuple")
    verb = op[0]

    def ok_ref(ref: Any) -> None:
        if _is_index(ref):
            return
        # A string is not a path. Unchecked, it would resolve to the ROOT and write there.
        if not isinstance(ref, (list, tuple)):
            raise TypeError("path is not an array")
        assert_safe_path(ref)

    if verb == "r":
        if len(op) != 2:
            raise TypeError("r arity")
        return
    if verb == "s":
        if len(op) == 3:
            ok_ref(op[1])
        elif len(op) != 2:
            raise TypeError("s arity")
        return
    if verb == "d":
        if len(op) == 2:
            ok_ref(op[1])
        elif len(op) != 1:
            raise TypeError("d arity")
        return
    if verb == "a":
        if len(op) == 3:
            ok_ref(op[1])
            if not isinstance(op[2], str):
                raise TypeError("a value")
        elif len(op) == 2:
            if not isinstance(op[1], str):
                raise TypeError("a value")
        else:
            raise TypeError("a arity")
        return
    if verb == "t":
        if len(op) == 3:
            ok_ref(op[1])
            if not _is_index(op[2]):
                raise TypeError("t count")
        elif len(op) == 2:
            if not _is_index(op[1]):
                raise TypeError("t count")
        else:
            raise TypeError("t arity")
        return
    if verb == "p":
        if len(op) == 5:
            ok_ref(op[1])
            index, remove, items = op[2], op[3], op[4]
        elif len(op) == 4:
            index, remove, items = op[1], op[2], op[3]
        else:
            raise TypeError("p arity")
        if not _is_index(index):
            raise TypeError("p index")
        if not _is_index(remove):
            raise TypeError("p remove")
        if not isinstance(items, list):
            raise TypeError("p items")
        return
    if verb == "m":
        if len(op) == 3:
            ok_ref(op[1])
        elif len(op) != 2:
            raise TypeError("m arity")
        _assert_permutation(op[-1])
        return
    if verb == "#":
        if len(op) != 3 or not _is_index(op[1]) or not isinstance(op[2], (list, tuple)):
            raise TypeError("# shape")
        assert_safe_path(op[2])
        return
    raise TypeError(f"unknown op verb: {verb!r}")


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


def is_base(ops: Sequence[Sequence[Any]]) -> bool:
    """A batch begins with a replacement: the recovery point for a fresh decoder."""
    return len(ops) > 0 and ops[0][0] == "r"


class WireEncoder:
    """Compresses decoded ops for one state stream: intern on SECOND use, short
    forms reuse the previous op's path within this batch. The id dictionary
    spans batches and an ``r`` clears it — a base batch is a recovery point, so
    everything after it must be self-contained."""

    def __init__(self) -> None:
        self._seen: Set[Tuple[Any, ...]] = set()
        self._ids: Dict[Tuple[Any, ...], int] = {}
        self._next_id = 0

    def encode(self, ops: Sequence[Sequence[Any]]) -> List[List[Any]]:
        previous: Any = None
        out: List[List[Any]] = []
        for op in ops:
            verb = op[0]
            if verb == "r":
                out.append(op)
                self._seen.clear()
                self._ids.clear()
                self._next_id = 0
                previous = None
                continue
            key = tuple(op[1])
            if key == previous:
                if verb == "s":
                    out.append(["s", op[2]])
                elif verb == "d":
                    out.append(["d"])
                elif verb == "a":
                    out.append(["a", op[2]])
                elif verb == "t":
                    out.append(["t", op[2]])
                elif verb == "p":
                    out.append(["p", op[2], op[3], op[4]])
                elif verb == "m":
                    out.append(["m", op[2]])
                else:
                    raise ValueError(f"unknown op verb: {verb!r}")
                continue
            ref: Any = list(op[1])
            existing = self._ids.get(key)
            if existing is not None:
                ref = existing
            elif key in self._seen:
                ref = self._next_id
                self._next_id += 1
                self._ids[key] = ref
                out.append(["#", ref, list(op[1])])
            else:
                self._seen.add(key)
            if verb == "s":
                out.append(["s", ref, op[2]])
            elif verb == "d":
                out.append(["d", ref])
            elif verb == "a":
                out.append(["a", ref, op[2]])
            elif verb == "t":
                out.append(["t", ref, op[2]])
            elif verb == "p":
                out.append(["p", ref, op[2], op[3], op[4]])
            elif verb == "m":
                out.append(["m", ref, op[2]])
            else:
                raise ValueError(f"unknown op verb: {verb!r}")
            previous = key
        return out


class WireDecoder:
    """Expands wire ops for one state stream. Path ids span batches; short forms
    are scoped to the batch. Decoding does not copy payloads: fanning one decoded
    batch out to several replicas aliases their op payloads."""

    def __init__(self) -> None:
        self._paths: Dict[int, Any] = {}

    def decode(self, wire: Sequence[Sequence[Any]]) -> List[List[Any]]:
        previous: Any = None
        out: List[List[Any]] = []
        for op in wire:
            assert_valid_wire_op(op)
            verb = op[0]
            if verb == "#":
                self._paths[op[1]] = op[2]
                continue
            if verb == "r":
                out.append(op)
                self._paths.clear()
                previous = None
                continue
            # Arity tells us whether a ref is present: the short forms omit it.
            short = (verb == "d" and len(op) == 1) or (verb not in ("d", "p") and len(op) == 2) or (
                verb == "p" and len(op) == 4
            )
            if short:
                if previous is None:
                    raise PathError([])
                path = previous
            else:
                ref = op[1]
                if _is_index(ref):
                    path = self._paths.get(ref)
                    if path is None:
                        raise PathError(ref)
                else:
                    path = ref
                previous = path
            if verb not in ("p", "m") and len(path) == 0:
                raise PathError(path)
            if verb == "s":
                out.append(["s", path, op[1] if short else op[2]])
            elif verb == "d":
                out.append(["d", path])
            elif verb == "a":
                out.append(["a", path, op[1] if short else op[2]])
            elif verb == "t":
                out.append(["t", path, op[1] if short else op[2]])
            elif verb == "p":
                if short:
                    out.append(["p", path, op[1], op[2], op[3]])
                else:
                    out.append(["p", path, op[2], op[3], op[4]])
            elif verb == "m":
                out.append(["m", path, op[1] if short else op[2]])
            else:
                raise TypeError(f"unknown op verb: {verb!r}")
        return out


class RevisionValidator:
    """Validates replica revisions are strict JSON (mirror of ``JsonRevisionValidator``).

    The TS cross-revision container cache relies on JS WeakSet of objects;
    Python dict/list instances do not support weak references, so this port
    revalidates the whole revision each time."""

    def validate(self, value: Any) -> Any:
        self._validate(value, set(), set())
        return value

    def _validate(self, value: Any, ancestors: Set[int], finished: Set[int]) -> None:
        if not isinstance(value, (dict, list)):
            if value is None or isinstance(value, (str, bool)):
                return
            if isinstance(value, int) and not isinstance(value, bool):
                return
            if isinstance(value, float) and math.isfinite(value):
                return
            raise TypeError("Replicated state values must be strict JSON")
        if id(value) in finished:
            return
        if id(value) in ancestors:
            raise TypeError("Replicated state cannot contain cycles")
        ancestors.add(id(value))
        try:
            if isinstance(value, list):
                for item in value:
                    self._validate(item, ancestors, finished)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("Replicated state objects must contain string keys")
                    self._validate(item, ancestors, finished)
            finished.add(id(value))
            return
        finally:
            ancestors.discard(id(value))
