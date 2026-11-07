"""Durable tool contracts (compact port of ``harness/define.ts`` and the tool
parts of ``harness/types.ts``).

The TS ``ToolExecutionApi`` is a large host object; this port keeps the parts
the durable tools use: the execution environment, running output, the output
window, and diagnostics. Task/document operations are not ported yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..env import ExecutionEnv, ShellOutputSkip


@dataclass
class ToolDiagnostic:
    """Remark about a call for the model and the UI; never part of the tool's data."""

    severity: str  # "info" | "warn" | "error"
    message: str
    code: Optional[str] = None


@dataclass
class ToolExecutionResult:
    """What one tool invocation reports. Omitted content falls back to the retained ``output()`` text."""

    content: Optional[list] = None  # [{"type": "text", "text": str} | {"type": "image", "data": base64 str, "mime_type": str}] blocks
    is_error: bool = False
    details: Optional[dict] = None
    diagnostics: Optional[list] = None  # extra ToolDiagnostic beyond api.diagnostic()


class ToolExecutionApi:
    """Operations available to one tool invocation. A plain object a host passes per call."""

    env: Optional[ExecutionEnv] = None
    output_window: Optional[Any] = None  # ShellOutputWindow | None

    def output(self, chunk: str, skipped: Optional[ShellOutputSkip] = None) -> None:
        """Append running output; it becomes the result content when the result omits ``content``."""
        raise NotImplementedError

    def diagnostic(self, diagnostic: ToolDiagnostic) -> None:
        """Record a model-visible remark about this call."""
        raise NotImplementedError


@dataclass
class ToolRegistration:
    """Executable tool registered in a registry. Only the pi-ai ``Tool`` fields enter the transcript."""

    name: str
    description: str
    parameters: dict  # JSON schema
    execute: Callable[[dict, ToolExecutionApi, Any], Awaitable[ToolExecutionResult]]
    prepare_arguments: Optional[Callable[[Any], dict]] = None
    output_limits: Optional[dict] = None  # {"maxBytes"?, "maxLines"?, "retain"?: "head"|"tail"}

    @property
    def label(self) -> str:
        """pi-ai Tool surface alias."""
        return self.name


def define_tool(**kwargs: Any) -> ToolRegistration:
    """Identity function that types a tool (kept for parity with ``defineTool``)."""
    return ToolRegistration(**kwargs)


def require_env(api: ToolExecutionApi) -> ExecutionEnv:
    """The call's execution environment; a tool without one fails with an ordinary error."""
    if api.env is None:
        raise RuntimeError("No execution environment is configured")
    return api.env
