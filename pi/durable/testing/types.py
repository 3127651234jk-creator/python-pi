"""Conformance assertions and case types (port of ``testing/types.ts`` and
``testing/assertions.ts``).

The suites are runner-independent: ``create_*_conformance`` returns plain case
objects, and a test module drives them with whatever runner it has (pytest
helpers live in ``runner.py``).
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional


class ConformanceAssertions:
    """The assertion vocabulary both conformance suites use."""

    def ok(self, value: Any, message: Optional[str] = None) -> None:
        assert value, message or "Expected a truthy value"

    def strict_equal(self, actual: Any, expected: Any) -> None:
        assert actual == expected, f"Expected {expected!r}, got {actual!r}"

    def deep_equal(self, actual: Any, expected: Any) -> None:
        assert actual == expected, f"Expected {expected!r}, got {actual!r}"

    def partial_deep_equal(self, actual: Any, expected: Any) -> None:
        _assert_partial(actual, expected, [])

    def greater_than(self, actual: float, expected: float) -> None:
        assert actual > expected, f"Expected {actual!r} > {expected!r}"

    async def rejects(self, operation: Awaitable[Any], message_includes: str) -> None:
        try:
            await asyncio.ensure_future(operation)
        except BaseException as exc:  # noqa: BLE001 - the rejection itself is the assertion target
            if message_includes and message_includes not in str(exc):
                raise AssertionError(
                    f"Expected an error containing {message_includes!r}, got {type(exc).__name__}: {exc}"
                ) from exc
            return
        raise AssertionError(f"Expected an error containing {message_includes!r}, but the operation resolved")


def _assert_partial(actual: Any, expected: Any, where: List[Any]) -> None:
    if isinstance(expected, dict):
        if isinstance(actual, dict):
            fields = actual
        elif dataclasses.is_dataclass(actual) and not isinstance(actual, type):
            fields = {field.name: getattr(actual, field.name) for field in dataclasses.fields(actual)}
        else:
            raise AssertionError(f"At {where}: expected an object, got {type(actual).__name__}: {actual!r}")
        for key, sub in expected.items():
            if key not in fields:
                raise AssertionError(f"At {where + [key]}: missing key {key!r}")
            _assert_partial(fields[key], sub, where + [key])
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"At {where}: expected an array, got {type(actual).__name__}: {actual!r}")
        for i, sub in enumerate(expected):
            _assert_partial(actual[i], sub, where + [i])
        return
    if actual != expected:
        raise AssertionError(f"At {where}: expected {expected!r}, got {actual!r}")


@dataclass
class StorageConformanceCase:
    name: str
    run: Callable[[], Awaitable[None]]


@dataclass
class EnvConformanceCase:
    name: str
    run: Callable[[], Awaitable[None]]
    # Cases that wait for an environment's watch latency need longer than a test runner's default timeout.
    timeout_seconds: Optional[float] = None


StorageConformanceProvider = Callable[[Callable[[Any], Awaitable[None]]], Awaitable[None]]
EnvConformanceProvider = Callable[[Callable[[Any], Awaitable[None]]], Awaitable[None]]


@dataclass
class StorageConformanceOptions:
    assertions: ConformanceAssertions
    with_storage: StorageConformanceProvider


@dataclass
class EnvConformanceOptions:
    assertions: ConformanceAssertions
    with_env: EnvConformanceProvider
    # Program and flag that run a POSIX shell script from the next argument; default ["sh", "-c"].
    shell: Optional[List[str]] = None
    # Whether the shell's ``ln -s`` creates symbolic links; default true.
    symlinks: Optional[bool] = None
