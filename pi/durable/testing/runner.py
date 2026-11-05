"""Pytest integration for the conformance suites (port of ``testing/runner.ts``).

The TS runner registers cases with Vitest/Jest; the Python port generates
parametrized pytest tests. ``make_storage_conformance_tests`` and
``make_env_conformance_tests`` create module-level pytest functions from a
provider, so each case runs as its own test.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from .env_conformance import create_env_conformance
from .storage_conformance import create_storage_conformance
from .types import ConformanceAssertions, EnvConformanceOptions, StorageConformanceOptions


def make_storage_conformance_tests(
    name: str,
    with_storage: Callable,
    backends: Optional[List[Tuple[str, Callable[[], Any]]]] = None,
    timeout_seconds: float = 60.0,
) -> List[Dict[str, Any]]:
    """Create one pytest case descriptor per conformance case.

    Returns ``{"id", "test"}`` items; a test module turns them into pytest
    tests (see ``tests/test_durable_storage_conformance.py``).
    """
    options = StorageConformanceOptions(assertions=ConformanceAssertions(), with_storage=with_storage)
    cases = create_storage_conformance(options)
    tests: List[Dict[str, Any]] = []
    for case in cases:
        tests.append({"id": f"{name}[{case.name}]" if backends is None else case.name, "run": case.run})
    return tests


def make_env_conformance_tests(
    with_env: Callable,
    shell: Optional[List[str]] = None,
    symlinks: Optional[bool] = None,
    default_timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Create pytest case descriptors from ``create_env_conformance``."""
    options = EnvConformanceOptions(
        assertions=ConformanceAssertions(), with_env=with_env, shell=shell, symlinks=symlinks
    )
    cases = create_env_conformance(options)
    return [
        {
            "id": case.name,
            "run": case.run,
            "timeout": case.timeout_seconds or default_timeout,
        }
        for case in cases
    ]


def run_case(run: Callable[[], Any]) -> None:
    """Run one case's coroutine under a fresh event loop."""
    asyncio.run(run())
