"""Program Status Protocol (OSC 7501, F-TUI-01): the running pi reports idle /
working / blocked / done / error to terminals that support it.

Port of ``packages/tui/src/program-status.ts`` (protocol) and
``packages/coding-agent/src/modes/interactive/program-status-reporter.ts``
(state wiring). Only the root record is supported.

Spec: https://www.superlogical.com/rex/docs/build/program-status
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Feature-detection query; a supporting terminal replies with the same body.
PROGRAM_STATUS_QUERY = "\x1b]7501;?\x1b\\"

APP_NAME = "pi"
APP_PATTERN = re.compile(r"^[A-Za-z0-9_.+-]{1,32}$")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
# Decoded `msg` limit; its base64 encoding stays under the wire limit.
MAX_MESSAGE_BYTES = 2048

STATES = ("idle", "working", "blocked", "done", "error", "clear")
KINDS = ("permission", "question", "auth")


def is_program_status_reply(sequence: str) -> bool:
    """The reply to PROGRAM_STATUS_QUERY; later revisions may add pairs."""
    return re.fullmatch(r"\x1b]7501;\?[^\x07\x1b]*(?:\x07|\x1b\\)", sequence) is not None


def truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    kept: list = []
    size = 0
    for char in text:
        width = len(char.encode("utf-8"))
        if size + width > max_bytes:
            break
        kept.append(char)
        size += width
    return "".join(kept)


@dataclass
class ProgramStatus:
    state: str
    app: Optional[str] = None
    kind: Optional[str] = None
    message: Optional[str] = None


def format_program_status(status: ProgramStatus) -> str:
    """Encode a status report. Control characters become spaces so terminals
    that discard such text still accept the message."""
    pairs = [f"state={status.state}"]
    if status.app is not None and APP_PATTERN.match(status.app):
        pairs.append(f"app={status.app}")
    if status.state == "blocked" and status.kind:
        pairs.append(f"kind={status.kind}")
    message = truncate_utf8(CONTROL_CHARACTERS.sub(" ", (status.message or "")).strip(), MAX_MESSAGE_BYTES)
    if message:
        pairs.append("msg=" + base64.b64encode(message.encode("utf-8")).decode("ascii"))
    return f"\x1b]7501;{':'.join(pairs)}\x1b\\"


def program_status_enabled() -> bool:
    """Whether to report: ``PI_PROGRAM_STATUS=1`` opts in and ``=0`` opts out.

    The TS startup probe writes the query and reads the raw reply; Python's
    input stack (prompt_toolkit / line editing) has no portable raw window for
    that handshake, so without the override the feature stays off.
    """
    value = os.environ.get("PI_PROGRAM_STATUS")
    if value == "1":
        return True
    if value == "0":
        return False
    return False


def first_line(text: Optional[str]) -> str:
    if not text:
        return "Error"
    return text.splitlines()[0].strip() if text.splitlines() else "Error"


class ProgramStatusReporter:
    """Maps agent events and dialogs to terminal status reports.

    ``emit`` receives the OSC sequence (an output writer); reports are
    deduplicated so the terminal only sees changes.
    """

    def __init__(self, emit: Callable[[str], None], get_session_name: Callable[[], Optional[str]]) -> None:
        self.emit = emit
        self.get_session_name = get_session_name
        self.run_active = False
        self.compacting = False
        self.run_result: ProgramStatus = ProgramStatus(state="done")
        self.resting_status: ProgramStatus = ProgramStatus(state="idle")
        self.blocked: Dict[str, Tuple[str, str]] = {}
        self.last_report: Optional[str] = None

    def handle_event(self, event) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "agent_start":
            self.run_active = True
            self.run_result = ProgramStatus(state="done")
        elif event_type == "message_end":
            message = getattr(event, "message", None)
            if getattr(message, "role", None) != "assistant":
                return
            if getattr(message, "stop_reason", None) == "error":
                self.run_result = ProgramStatus(state="error", message=first_line(message.error_message))
            else:
                self.run_result = ProgramStatus(state="done")
        elif event_type == "compaction_start":
            self.compacting = True
        elif event_type == "compaction_end":
            self.compacting = False
            if self.run_active:
                if getattr(event, "aborted", False):
                    self.run_result = ProgramStatus(state="idle")
                elif getattr(event, "error_message", None):
                    self.run_result = ProgramStatus(state="error", message=first_line(event.error_message))
            elif getattr(event, "aborted", False):
                self.resting_status = ProgramStatus(state="idle")
            elif getattr(event, "reason", None) == "manual":
                self.resting_status = (ProgramStatus(state="error", message=first_line(event.error_message))
                                       if getattr(event, "error_message", None) else ProgramStatus(state="done"))
        elif event_type == "agent_end":
            # The agent's final end is the settle point unless a retry
            # supersedes it; Python has no separate agent_settled agent event.
            if getattr(event, "will_retry", False):
                return
            self.run_active = False
            messages = getattr(event, "messages", None) or []
            final = next(
                (message for message in reversed(messages)
                 if getattr(message, "role", None) == "assistant"),
                None,
            )
            aborted = final is not None and getattr(final, "stop_reason", None) == "aborted"
            self.resting_status = ProgramStatus(state="idle") if aborted else self.run_result
        elif event_type == "agent_settled":
            self.run_active = False
            aborted = bool(getattr(event, "aborted", False))
            self.resting_status = ProgramStatus(state="idle") if aborted else self.run_result
        else:
            return
        self.report()

    def set_blocked(self, source: str, status: Optional[Tuple[str, str]]) -> None:
        """Report `blocked` for a dialog until cleared with None; reopening a
        source replaces it. The most recent dialog wins."""
        self.blocked.pop(source, None)
        if status is not None:
            self.blocked[source] = status
        self.report()

    def reset(self) -> None:
        self.run_active = False
        self.compacting = False
        self.run_result = ProgramStatus(state="done")
        self.resting_status = ProgramStatus(state="idle")
        self.report()

    def report(self) -> None:
        status = self._current_status()
        status.app = APP_NAME
        key = json.dumps(status.__dict__, sort_keys=True)
        if key == self.last_report:
            return
        self.last_report = key
        self.emit(format_program_status(status))

    def _current_status(self) -> ProgramStatus:
        blocked = None
        for value in self.blocked.values():
            blocked = value
        if blocked is not None:
            return ProgramStatus(state="blocked", kind=blocked[0], message=blocked[1])
        if self.compacting:
            return ProgramStatus(state="working", message="Compacting context")
        status = ProgramStatus(state="working") if self.run_active else self.resting_status
        if status.state in ("working", "done"):
            status.message = self.get_session_name()
        return status
