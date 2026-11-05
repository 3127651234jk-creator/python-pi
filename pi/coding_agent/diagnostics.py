"""Startup diagnostics: one shaped type for everything the CLI reports.

Port of ``diagnostics.ts`` and ``settings-diagnostics.ts``. Producers append
``Diagnostic`` items (settings load failures, resource discovery warnings,
extension and MCP failures); the CLI deduplicates and reports them once.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class Diagnostic:
    type: str  # "warning" | "error" | "collision"
    message: str
    path: Optional[str] = None


def settings_diagnostics(errors: Iterable) -> List[Diagnostic]:
    """One diagnostic per collected settings-file error: (path, message) pairs."""
    return [Diagnostic("warning", f"Invalid settings file {path}: {message}" if path
                       else f"Invalid merged settings: {message}", path=str(path) or None)
            for path, message in errors]


def deduplicate_diagnostics(diagnostics: Iterable[Diagnostic]) -> List[Diagnostic]:
    seen = set()
    result = []
    for diagnostic in diagnostics:
        key = (diagnostic.type, diagnostic.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return result


def report_diagnostics(diagnostics: Iterable[Diagnostic]) -> None:
    import sys

    for diagnostic in diagnostics:
        prefix = "error: " if diagnostic.type == "error" else "warning: "
        print(prefix + diagnostic.message, file=sys.stderr)
