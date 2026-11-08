"""Codemode: organize, compose and execute tool calls through Python code.

Port of the codemode tool from ``packages/codemode`` and the coding-agent
extension: the model writes a Python script that calls the agent's tools
programmatically instead of one tool call at a time. The script runs in a fresh
``python -I`` subprocess that can only reach the outside world through a
JSON-line protocol to the host (:mod:`pi.codemode.host`), which executes tool
calls through the agent's normal pipeline (so validation, hooks, and permission
checks apply). No network/process/timer primitives are provided; isolation is
process-level.

Modules:
  tool          agent-tool adapter: declaration, description, store, output
  host          subprocess lifecycle, line protocol, call records
  worker_source the embedded worker program, run with ``python -I``
"""

from .tool import create_codemode_tool

__all__ = ["create_codemode_tool"]
