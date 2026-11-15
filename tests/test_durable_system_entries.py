"""pi.system entry model: the durable prompt state lives in the transcript.

Generation prepare plans positional pi.system entries (sections + tool
declarations) against the state replayed from committed entries: a full
baseline when the active range has none, a minimal diff otherwise, and omit
edits for entries retained across a compaction. Regression coverage for the
B-AG-01 alignment (TS #10542 semantics) and the prompt.py helpers.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pi.durable.harness as harness_module
from pi.ai.json_io import to_json
from pi.ai.models import Model
from pi.ai.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    Tool,
    Usage,
    UserMessage,
    get_message_text,
    get_system_message_text,
)
from pi.ai.utils import get_current_tools, normalize_context, TranscriptContext
from pi.agent.types import AgentTool, AgentToolResult
from pi.durable.entries import COMPACTION_ENTRY, SYSTEM_ENTRY
from pi.durable.extensions import Extension
from pi.durable.harness import Harness
from pi.durable.prompt import INSTRUCTIONS_KEY, lead_with_system, plan_system_entries
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage
from pi.durable.tasks import compact_definition
from pi.durable.types import EntryRecord


def make_model(context_window=None) -> Model:
    model = Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")
    if context_window is not None:
        model.context_window = context_window
    return model


def answer_message(text: str = "answer") -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], api="faux", provider="faux",
                            model="faux", usage=Usage(), stop_reason="stop", timestamp=0)


def section_extension(name: str = "style-ext", key: str = "style", text: str = "Use a terse style."):
    async def render(api, runtime):
        return text

    return Extension(name, sections={key: render})


def echo_tool_extension(name: str = "tool-ext", tool_name: str = "echo"):
    async def execute(call_id, arguments, signal, update):
        return AgentToolResult(content=[TextContent(text="echoed")])

    return Extension(name, tools=[AgentTool(name=tool_name, description="echo", execute=execute)])


def make_harness(tmp_path, *extensions, **kwargs):
    session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
    harness = Harness(session, make_model(), extensions=list(extensions), **kwargs)
    return harness, session


async def run_turn(harness, cid, text="hello", recorder=None):
    """Run one generation with a canned stream; returns the request messages."""

    calls = []

    async def stream(model, messages, harness_arg, **kwargs):
        calls.append(list(messages))
        return answer_message()

    original = harness_module._stream_once
    harness_module._stream_once = stream
    try:
        await harness.submit_input(cid, text)
        await harness.run_pending_tasks()
    finally:
        harness_module._stream_once = original
    if recorder is not None:
        recorder.extend(calls)
    return calls


async def system_records(harness, cid):
    transcript = await harness.transcript(cid)
    return [entry for entry in transcript if entry.kind == SYSTEM_ENTRY.kind]


def folded_system(messages):
    system, _ = normalize_context(TranscriptContext(messages=messages))
    return system


# ---------------------------------------------------------------------------
# baseline and idempotence
# ---------------------------------------------------------------------------


def test_first_generation_writes_baseline_and_second_is_idempotent(tmp_path):
    async def scenario():
        harness, session = make_harness(tmp_path)
        cid = await harness.create_conversation()

        calls = await run_turn(harness, cid, "first")

        entries = await system_records(harness, cid)
        assert len(entries) == 1
        assert entries[0].edits is None
        # No extensions, no instructions: the baseline establishes empty state.
        assert entries[0].model[0].role == "system"

        # The request context is the entry-order messages with the system led.
        assert [message.role for message in calls[0]] == ["system", "user"]

        await run_turn(harness, cid, "second")
        entries = await system_records(harness, cid)
        assert len(entries) == 1  # no diff, no new entry
        await session.close()

    asyncio.run(scenario())


def test_instructions_change_writes_minimal_diff(tmp_path):
    async def scenario():
        harness, session = make_harness(tmp_path, section_extension())
        cid = await harness.create_conversation()
        await harness.configure(cid, instructions="be terse")

        await run_turn(harness, cid, "first")
        entries = await system_records(harness, cid)
        assert len(entries) == 1
        assert entries[0].model[0].sections == {"style": "Use a terse style.", INSTRUCTIONS_KEY: "be terse"}

        await harness.configure(cid, instructions="be verbose")
        requests = await run_turn(harness, cid, "second")
        entries = await system_records(harness, cid)
        assert len(entries) == 2
        patch = entries[1].model[0]
        assert patch.sections == {INSTRUCTIONS_KEY: "be verbose"}  # style untouched, not re-sent
        system = folded_system(requests[0])
        assert system.sections == {"style": "Use a terse style.", INSTRUCTIONS_KEY: "be verbose"}
        assert "be verbose" in get_system_message_text(system)
        await session.close()

    asyncio.run(scenario())


def test_section_rendering_none_omits_the_section(tmp_path):
    async def scenario():
        async def render(api, runtime):
            return None

        harness, session = make_harness(tmp_path, Extension("none-ext", sections={"skills": render}))
        cid = await harness.create_conversation()

        requests = await run_turn(harness, cid, "first")
        entries = await system_records(harness, cid)
        assert entries[0].model[0].sections is None  # omitted, never the string "None"
        system = folded_system(requests[0])
        assert system is None or "None" not in get_system_message_text(system)
        await session.close()

    asyncio.run(scenario())


def test_section_order_change_rewrites_entries(tmp_path):
    """A reordered section list cannot be patched in place; delete-all then add-all."""

    async def scenario():
        harness, session = make_harness(tmp_path)
        cid = await harness.create_conversation()
        await harness.configure(cid, instructions="old")

        await run_turn(harness, cid, "first")
        assert len(await system_records(harness, cid)) == 1

        # A second extension whose section must sort before the instructions key.
        harness.registry.install(section_extension("alpha-ext", key="alpha", text="alpha text"))
        await harness.configure(cid, extensions=["alpha-ext"], instructions="new")
        requests = await run_turn(harness, cid, "second")

        entries = await system_records(harness, cid)
        assert len(entries) == 3  # baseline, delete-all, add-all
        delete_entry = entries[1].model[0]
        add_entry = entries[2].model[0]
        assert set(delete_entry.sections.keys()) == {INSTRUCTIONS_KEY}
        assert list(add_entry.sections.keys()) == ["alpha", INSTRUCTIONS_KEY]

        system = folded_system(requests[0])
        assert list(system.sections.keys()) == ["alpha", INSTRUCTIONS_KEY]
        await session.close()

    asyncio.run(scenario())


def test_tool_changes_declared_through_entries(tmp_path):
    async def scenario():
        harness, session = make_harness(tmp_path, echo_tool_extension())
        cid = await harness.create_conversation()

        requests = await run_turn(harness, cid, "first")
        assert get_current_tools(requests[0])[0].name == "echo"
        baseline = (await system_records(harness, cid))[0].model[0]
        assert [tool.name for tool in baseline.tools_added] == ["echo"]

        await harness.configure(cid, extensions=[])
        requests = await run_turn(harness, cid, "second")
        entries = await system_records(harness, cid)
        assert len(entries) == 2
        patch = entries[1].model[0]
        assert [ref.name for ref in patch.tools_removed] == ["echo"]
        assert get_current_tools(requests[0]) == []
        await session.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# tool loadout order (TS planTools)
# ---------------------------------------------------------------------------


def _tool_entry_entries(tools):
    return [
        EntryRecord(id=1, conversation_id=1, kind="pi.user", model=[]),
        EntryRecord(id=2, conversation_id=1, kind=SYSTEM_ENTRY.kind,
                    model=[SystemMessage(content="", timestamp=0, tools_added=list(tools))]),
        EntryRecord(id=3, conversation_id=1, kind="pi.assistant", model=[]),
    ]


def test_tool_order_change_rewrites_the_loadout():
    a, b, c = (Tool(name=name, description=name, parameters={}) for name in ("a", "b", "c"))
    drafts, messages = plan_system_entries(_tool_entry_entries([a, b, c]), {}, [c, b, a])

    assert len(drafts) == 1
    assert [tool.name for tool in get_current_tools(messages)] == ["c", "b", "a"]


def test_changed_declaration_keeps_the_desired_tool_order():
    a, b, c = (Tool(name=name, description=name, parameters={}) for name in ("a", "b", "c"))
    a2 = Tool(name="a", description="a v2", parameters={})
    drafts, messages = plan_system_entries(_tool_entry_entries([a, b, c]), {}, [a2, b, c])

    assert len(drafts) == 1
    assert [tool.name for tool in get_current_tools(messages)] == ["a", "b", "c"]
    assert get_current_tools(messages)[0].description == "a v2"


# ---------------------------------------------------------------------------
# compaction / reset / fork
# ---------------------------------------------------------------------------


def test_compaction_omits_retained_system_and_rebaselines(tmp_path, monkeypatch):
    async def stream(model, messages, harness_arg, **kwargs):
        return answer_message("summary")

    monkeypatch.setattr("pi.durable.harness._stream_once", stream)

    async def scenario():
        harness, session = make_harness(tmp_path)
        harness.model = make_model(context_window=1000)
        harness.register_definition(compact_definition(keep_recent_tokens=1, reserve_tokens=1))
        cid = await harness.create_conversation()

        # Oversized context so the compaction must cut. The manual pi.system
        # write is the transcript's last entry, so the token walk stops on it:
        # the cut keeps it, and the next generation re-baselines behind the
        # compaction marker and omits it.
        await run_turn(harness, cid, "x" * 4000)
        await harness.configure(cid, instructions="old rules")
        await harness.submit_write(cid, {
            "kind": SYSTEM_ENTRY.kind,
            "model": [to_json(SystemMessage(content="", timestamp=0,
                                            sections={INSTRUCTIONS_KEY: "old rules"}))],
        })
        await harness.run_pending_tasks()
        manual = (await system_records(harness, cid))[-1].id  # submit_write returns a submission id

        compact_id = await harness.submit_compact_task(cid)
        await harness.run_pending_tasks()
        record = await harness._get_task(compact_id)
        assert record.state.outcome.status == "completed"
        assert record.state.outcome.result["compacted"] is True
        compaction_entry = (await harness.transcript(cid))[-1]
        assert compaction_entry.kind == COMPACTION_ENTRY.kind
        assert compaction_entry.head == manual  # a cut may keep a trailing system entry

        requests = await run_turn(harness, cid, "after compaction")
        entries = await system_records(harness, cid)
        assert len(entries) == 2
        rebaseline = entries[1]
        assert rebaseline.model[0].sections == {INSTRUCTIONS_KEY: "old rules"}
        targets = [edit.target for edit in rebaseline.edits or []]
        assert manual in targets  # the retained manual baseline is omitted

        system = folded_system(requests[0])
        assert system.sections == {INSTRUCTIONS_KEY: "old rules"}
        assert get_current_tools(requests[0]) == []
        await session.close()

    asyncio.run(scenario())


def test_compaction_still_cuts_when_the_system_entry_exceeds_keep_recent(tmp_path, monkeypatch):
    """A cut that lands on a system entry must not abort the compaction."""

    async def scenario():
        harness, session = make_harness(tmp_path)
        harness.model = make_model(context_window=1000)
        harness.register_definition(compact_definition(keep_recent_tokens=500, reserve_tokens=1))
        cid = await harness.create_conversation()

        # The baseline's instructions alone exceed keep_recent_tokens, so the
        # token walk stops on that system entry.
        await harness.configure(cid, instructions="rule " * 1500)
        await run_turn(harness, cid, "x " * 3000)
        baseline = (await system_records(harness, cid))[0]

        async def summarize_stream(model, messages, *args, **kwargs):
            return answer_message("summary")

        # tasks.py binds _stream_once at import time, so patch its module global.
        monkeypatch.setattr("pi.durable.tasks._stream_once", summarize_stream)
        compact_id = await harness.submit_compact_task(cid)
        await harness.run_pending_tasks()
        record = await harness._get_task(compact_id)
        assert record.state.outcome.result["compacted"] is True
        compaction_entry = (await harness.transcript(cid))[-1]
        assert compaction_entry.head == baseline.id
        await session.close()

    asyncio.run(scenario())


def test_compaction_summary_skips_omitted_entries(tmp_path, monkeypatch):
    """Suppressed prompt text must not count toward compaction or its summary."""

    async def scenario():
        harness, session = make_harness(tmp_path)
        harness.model = make_model(context_window=1000)
        harness.register_definition(compact_definition(keep_recent_tokens=1, reserve_tokens=1))
        cid = await harness.create_conversation()

        await run_turn(harness, cid, "x " * 3000)
        await harness.submit_write(cid, {
            "kind": SYSTEM_ENTRY.kind,
            "model": [to_json(SystemMessage(content="", timestamp=0,
                                            sections={"leak": "SECRET-OMITTED"}))],
        })
        leak_id = (await system_records(harness, cid))[-1].id
        await harness.submit_write(cid, {
            "kind": SYSTEM_ENTRY.kind,
            "model": [],
            "edits": [{"target": leak_id, "action": "omit"}],
        })
        await run_turn(harness, cid, "y " * 3000)

        requests = []

        async def capture(model, messages, *args, **kwargs):
            requests.append(list(messages))
            return answer_message("summary")

        # tasks.py binds _stream_once at import time, so patch its module global.
        monkeypatch.setattr("pi.durable.tasks._stream_once", capture)
        compact_id = await harness.submit_compact_task(cid)
        await harness.run_pending_tasks()
        compact_record = await harness._get_task(compact_id)
        assert compact_record.state.outcome.result is not None, compact_record.state.outcome
        assert compact_record.state.outcome.result["compacted"] is True
        assert requests, "the summarizer was not called"
        summary_source = "".join(get_message_text(message.content) for message in requests[0])
        assert "SECRET-OMITTED" not in summary_source
        await session.close()

    asyncio.run(scenario())


def test_reset_rebaselines_without_omit_edits(tmp_path):
    async def scenario():
        harness, session = make_harness(tmp_path)
        cid = await harness.create_conversation()
        await harness.configure(cid, instructions="first life")

        await run_turn(harness, cid, "first")
        old_baseline = (await system_records(harness, cid))[0]
        await harness.submit_write(cid, {"kind": "pi.reset", "head": "self", "model": []})
        await run_turn(harness, cid, "second")

        # The old baseline is out of the active range; the new one has no omit
        # edits because nothing retained needs suppressing.
        entries = await system_records(harness, cid)
        assert len(entries) == 1
        assert entries[0].id != old_baseline.id
        assert entries[0].edits is None
        assert entries[0].model[0].sections == {INSTRUCTIONS_KEY: "first life"}
        await session.close()

    asyncio.run(scenario())


def test_forked_conversation_replays_parent_system(tmp_path):
    async def scenario():
        harness, session = make_harness(tmp_path)
        cid = await harness.create_conversation()
        await harness.configure(cid, instructions="inherited rules")
        await run_turn(harness, cid, "parent turn")

        parent_baseline = (await system_records(harness, cid))[0]
        fork_at = parent_baseline.id  # inclusive: the baseline stays visible

        async def change(tx):
            return await tx.fork_conversation(cid, fork_at, {"kind": "ownerless"})

        child = (await session.commit(change)).id
        requests = await run_turn(harness, child, "child turn")

        # The parent's baseline is visible through the fork and replays as the
        # current state; the child plans no new entry.
        entries = await system_records(harness, child)
        assert len(entries) == 1
        assert entries[0].id == parent_baseline.id
        system = folded_system(requests[0])
        assert system.sections == {INSTRUCTIONS_KEY: "inherited rules"}
        await session.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# hooks and helpers
# ---------------------------------------------------------------------------


def test_before_request_sees_entry_system(tmp_path):
    async def scenario():
        seen = []

        async def observe(args, api, runtime):
            seen.append(list(args["messages"]))
            return None

        hook_ext = Extension("hook-ext", hooks={"before_request": observe})
        harness, session = make_harness(tmp_path, hook_ext)
        cid = await harness.create_conversation()
        await harness.configure(cid, instructions="hook rules")

        await run_turn(harness, cid, "hello")
        assert seen, "before_request hook did not run"
        messages = seen[0]
        assert messages[0].role == "system"
        assert messages[0].sections == {INSTRUCTIONS_KEY: "hook rules"}
        assert messages[1].role == "user"
        await session.close()

    asyncio.run(scenario())


def test_lead_with_system_moves_only_user_preceded_system():
    system = SystemMessage(content="s", timestamp=0)
    user = UserMessage(content="u", timestamp=0)
    assistant = answer_message()

    assert [m.role for m in lead_with_system([user, user, system, assistant])] == \
        ["system", "user", "user", "assistant"]
    assert lead_with_system([system, user]) == [system, user]  # already leading
    assert lead_with_system([user, assistant, system]) == [user, assistant, system]
    assert lead_with_system([user]) == [user]
    assert lead_with_system([]) == []


def test_normalize_context_deletes_none_sections():
    first = SystemMessage(content="", timestamp=0, sections={"keep": "one", "drop": "gone"})
    second = SystemMessage(content="", timestamp=0, sections={"drop": None, "add": "two"})
    system, rest = normalize_context(TranscriptContext(messages=[first, second]))
    assert system.sections == {"keep": "one", "add": "two"}
    assert rest == []
