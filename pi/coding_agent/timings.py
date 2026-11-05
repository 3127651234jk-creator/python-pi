"""Startup timing instrumentation, enabled with PI_TIMING=1.

Port of ``packages/coding-agent/src/core/timings.ts``: callers mark named
steps and ``print_timings`` writes one grouped report per namespace to stderr.
"""

import os
import sys
from time import perf_counter

ENABLED = os.environ.get("PI_TIMING") == "1"

_namespaces: dict = {}


def reset_timings(namespace: str = "main") -> None:
    if not ENABLED:
        return
    _namespaces[namespace] = {"timings": [], "last": perf_counter()}


def time(label: str, namespace: str = "main") -> None:
    if not ENABLED:
        return
    if namespace not in _namespaces:
        reset_timings(namespace)
    now = perf_counter()
    state = _namespaces[namespace]
    state["timings"].append((label, max(0, int((now - state["last"]) * 1000))))
    state["last"] = now


def _print_group(title: str, timings) -> None:
    printable = [entry for entry in timings if entry[1] >= 0]
    if not printable:
        return
    print(f"\n--- {title} ---", file=sys.stderr)
    for label, ms in printable:
        print(f"  {label}: {ms}ms", file=sys.stderr)
    print(f"  TOTAL: {sum(ms for _, ms in printable)}ms", file=sys.stderr)
    print("-" * (len(title) + 8) + "\n", file=sys.stderr)


def print_timings() -> None:
    if not ENABLED:
        return
    for namespace, state in _namespaces.items():
        _print_group(f"Startup Timings: {namespace}", state["timings"])
