"""OSC 7501 program status: encoding, event wiring, dialog blocking (F-TUI-01)."""

import asyncio
import base64
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.agent.agent import Agent
from pi.agent.types import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    MessageEndEvent,
)
from pi.ai.models import resolve_model
from pi.ai.providers import faux
from pi.ai.types import AssistantMessage, Usage
from pi.coding_agent.program_status import (
    PROGRAM_STATUS_QUERY,
    ProgramStatus,
    ProgramStatusReporter,
    format_program_status,
    is_program_status_reply,
    program_status_enabled,
)


def assistant(stop_reason="stop", error_message=None):
    return AssistantMessage(content=[], api="faux", provider="faux", model="faux",
                            usage=Usage(), stop_reason=stop_reason, timestamp=0,
                            error_message=error_message)


def settled(aborted):
    """agent_settled reaches the reporter as a wire-shaped payload."""
    return SimpleNamespace(type="agent_settled", aborted=aborted)


def test_format_program_status_encodes_pairs():
    assert format_program_status(ProgramStatus(state="idle")) == "\x1b]7501;state=idle\x1b\\"
    working = format_program_status(ProgramStatus(state="working", app="pi", message="deep dive"))
    assert "\x1b]7501;state=working:app=pi:msg=" in working
    assert base64.b64decode(working.split("msg=")[1].split("\x1b")[0]) == b"deep dive"
    blocked = format_program_status(ProgramStatus(state="blocked", app="pi", kind="question", message="Continue?"))
    assert ":kind=question:" in blocked
    # Control characters become spaces; invalid app names are dropped.
    sanitized = format_program_status(ProgramStatus(state="error", app="bad name!", message="a\x01b"))
    assert "app=" not in sanitized
    assert base64.b64decode(sanitized.split("msg=")[1].split("\x1b")[0]) == b"a b"
    assert is_program_status_reply("\x1b]7501;?\x1b\\") is True
    assert is_program_status_reply("\x1b]7501;?extra\x07") is True
    assert is_program_status_reply("\x1b]7501;state=idle\x1b\\") is False
    assert PROGRAM_STATUS_QUERY == "\x1b]7501;?\x1b\\"


def test_reporter_maps_run_lifecycle():
    reports = []
    reporter = ProgramStatusReporter(reports.append, lambda: "deep dive")
    reporter.handle_event(AgentStartEvent())
    assert reports[-1] == format_program_status(ProgramStatus(state="working", app="pi", message="deep dive"))
    reporter.handle_event(MessageEndEvent(message=assistant()))
    reporter.handle_event(settled(aborted=False))
    assert reports[-1].startswith("\x1b]7501;state=done")
    # An error response decides the settled outcome; only its first line reports.
    reporter.handle_event(AgentStartEvent())
    reporter.handle_event(MessageEndEvent(message=assistant("error", "HTTP 503 from gateway: busy\nline two")))
    reporter.handle_event(settled(aborted=False))
    encoded = reports[-1].split("msg=")[1].split("\x1b")[0]
    assert base64.b64decode(encoded) == b"HTTP 503 from gateway: busy"
    # Aborted runs go back to idle.
    reporter.handle_event(AgentStartEvent())
    reporter.handle_event(settled(aborted=True))
    assert reports[-1] == "\x1b]7501;state=idle:app=pi\x1b\\"


def test_reporter_settles_on_the_agents_final_end():
    """The agent emits agent_end, not agent_settled; only the final, non-retry
    end settles the run."""

    reports = []
    reporter = ProgramStatusReporter(reports.append, lambda: None)
    reporter.handle_event(AgentStartEvent())
    reporter.handle_event(MessageEndEvent(message=assistant()))
    reporter.handle_event(AgentEndEvent(messages=[assistant()], will_retry=False))
    assert reports[-1].startswith("\x1b]7501;state=done")

    # A retryable end does not settle: the run stays working.
    reporter.handle_event(AgentStartEvent())
    reporter.handle_event(MessageEndEvent(message=assistant("error", "HTTP 503")))
    reporter.handle_event(AgentEndEvent(messages=[assistant("error", "HTTP 503")], will_retry=True))
    assert "state=working" in reports[-1]

    # The final aborted turn returns to idle.
    reporter.handle_event(AgentEndEvent(messages=[assistant("aborted")], will_retry=False))
    assert reports[-1] == "\x1b]7501;state=idle:app=pi\x1b\\"


def test_reporter_blocks_on_dialogs_and_reports_compaction():
    reports = []
    reporter = ProgramStatusReporter(reports.append, lambda: None)
    reporter.set_blocked("dialog", ("question", "Continue?"))
    assert ":kind=question:" in reports[-1]
    reporter.set_blocked("picker", ("question", "Pick one"))
    assert base64.b64decode(reports[-1].split("msg=")[1].split("\x1b")[0]) == b"Pick one"
    reporter.set_blocked("dialog", None)
    # The most recent dialog wins.
    assert base64.b64decode(reports[-1].split("msg=")[1].split("\x1b")[0]) == b"Pick one"
    reporter.set_blocked("picker", None)
    assert "state=idle" in reports[-1]
    reporter.handle_event(CompactionStartEvent(reason="threshold"))
    assert "state=working" in reports[-1]
    assert base64.b64decode(reports[-1].split("msg=")[1].split("\x1b")[0]) == b"Compacting context"
    reporter.handle_event(CompactionEndEvent(reason="threshold", result={"summary": "s"}, aborted=False))
    assert "state=idle" in reports[-1]


def test_reporter_settles_a_real_agent_run():
    """The wiring subscribes to the agent's event stream, which ends in
    agent_end (there is no agent_settled event)."""

    async def check():
        os.environ["FAUX_SCRIPT"] = '[{"text": "ok"}]'
        faux.reset_script_index()
        try:
            reports = []
            reporter = ProgramStatusReporter(reports.append, lambda: None)
            agent = Agent(resolve_model("faux"))
            agent.subscribe(reporter.handle_event)
            await agent.prompt("hello")
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()
        assert reports[-1] == "\x1b]7501;state=done:app=pi\x1b\\"

    asyncio.run(check())


def test_enabled_flag_follows_the_override(monkeypatch):
    monkeypatch.delenv("PI_PROGRAM_STATUS", raising=False)
    assert program_status_enabled() is False
    monkeypatch.setenv("PI_PROGRAM_STATUS", "1")
    assert program_status_enabled() is True
    monkeypatch.setenv("PI_PROGRAM_STATUS", "0")
    assert program_status_enabled() is False
