"""CLI flag tests for the v1.0.4 parity surface: session selection, system
prompt overrides, resource switches, tool selection, export and print."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.ai.providers.faux import reset_script_index
from pi.ai.types import UserMessage
from pi.coding_agent import timings as cli_timings
from pi.coding_agent import cli
from pi.coding_agent.cli import async_main, build_parser, resolve_prompt_input, tool_selection_from_args
from pi.coding_agent.export_html import generate_html
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.session import SessionManager, export_session_file
from pi.coding_agent.config import Configuration


@pytest.fixture(autouse=True)
def agent_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "agent"))
    yield tmp_path


def run_cli(*args):
    reset_script_index()
    return asyncio.run(async_main(list(args)))


def latest_session(tmp_path) -> dict:
    """The most recent session file parsed into records."""
    sessions = tmp_path / ".pi" / "sessions"
    path = max(sessions.glob("*.jsonl"), key=os.path.getmtime)
    return {"path": path, "records": [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]}


def system_record(records) -> dict:
    return next(record for record in records
                if record.get("type") == "message" and record["message"].get("role") == "system")


# -- parser surface ---------------------------------------------------------

def test_parser_exposes_v104_flags():
    known = {option for action in build_parser()._actions for option in action.option_strings}
    for flag in ("--resume", "-r", "--session-id", "--fork", "--system-prompt", "--append-system-prompt",
                 "--no-builtin-tools", "-nbt", "--print", "-p", "--export", "--models", "--extension", "-e",
                 "--no-extensions", "-ne", "--skill", "--no-skills", "-ns", "--prompt-template",
                 "--no-prompt-templates", "-np", "--theme", "--use-theme", "--no-themes",
                 "--no-context-files", "-nc", "--verbose", "--offline", "--version", "-v"):
        assert flag in known, flag


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert re.match(r"\d+\.\d+\.\d+", capsys.readouterr().out.strip())


# -- tool selection ---------------------------------------------------------

def test_no_builtin_tools_drops_only_builtins():
    selection = tool_selection_from_args(
        build_parser().parse_args(["--no-builtin-tools"]))
    for name in ("read", "bash", "edit", "write", "grep", "find", "ls", "powershell"):
        assert not selection.keeps(name)
    assert selection.keeps("mcp__srv__tool") and selection.declares("mcp__srv__tool")
    assert selection.keeps("my_ext_tool") and selection.declares("my_ext_tool")


def test_no_builtin_tools_combines_with_exclude():
    selection = tool_selection_from_args(
        build_parser().parse_args(["--no-builtin-tools", "--exclude-tools", "codemode"]))
    assert not selection.keeps("read") and not selection.keeps("codemode")
    assert selection.keeps("mcp__srv__tool")


# -- system prompt ----------------------------------------------------------

def test_resolve_prompt_input_reads_files(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("from file", encoding="utf-8")
    assert resolve_prompt_input(str(path)) == "from file"
    assert resolve_prompt_input("literal text") == "literal text"


def test_system_prompt_flags_shape_the_prompt(agent_env):
    prompt_file = agent_env / "custom.txt"
    prompt_file.write_text("CUSTOM SYSTEM PROMPT", encoding="utf-8")
    assert run_cli("--provider", "faux", "--no-mcp", "--system-prompt", str(prompt_file),
                   "--append-system-prompt", "APPEND MARKER", "hi") == 0
    system = system_record(latest_session(agent_env)["records"])["message"]["content"]
    assert system.startswith("CUSTOM SYSTEM PROMPT")
    # A custom system prompt replaces the default tools/rules/docs sections.
    assert "<tools>" not in system and "<rules>" not in system and "<docs>" not in system
    assert "<addendum>\nAPPEND MARKER\n</addendum>" in system


def test_no_context_files_skips_project_instructions(agent_env):
    (agent_env / "AGENTS.md").write_text("PROJECT INSTRUCTION MARKER", encoding="utf-8")
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "with-ctx", "hi") == 0
    content = system_record(latest_session(agent_env)["records"])["message"]["content"]
    assert "PROJECT INSTRUCTION MARKER" in content
    assert run_cli("--provider", "faux", "--no-mcp", "--no-context-files",
                   "--session-id", "without-ctx", "hi") == 0
    content = system_record(latest_session(agent_env)["records"])["message"]["content"]
    assert "PROJECT INSTRUCTION MARKER" not in content


# -- session selection ------------------------------------------------------

def test_session_id_creates_then_resumes(agent_env, capsys):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "my-task", "one") == 0
    assert "creating a new session" in capsys.readouterr().err
    created = agent_env / ".pi" / "sessions" / "my-task.jsonl"
    assert created.exists()
    before = len(created.read_text(encoding="utf-8").splitlines())
    capsys.readouterr()
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "my-task", "two") == 0
    assert "creating a new session" not in capsys.readouterr().err
    assert len(created.read_text(encoding="utf-8").splitlines()) > before


def test_session_id_rejects_invalid_and_conflicts(agent_env, capsys):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", ".bad", "hi") == 2
    assert "must start and end with a letter or digit" in capsys.readouterr().err
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "x", "--session", "y", "hi") == 2
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "x", "--no-session", "hi") == 2


def test_fork_flag_copies_the_source_session(agent_env):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "source", "one") == 0
    source = latest_session(agent_env)["records"]
    assert run_cli("--provider", "faux", "--no-mcp", "--fork", "source",
                   "--session-id", "forked", "two") == 0
    forked = agent_env / ".pi" / "sessions" / "forked.jsonl"
    records = [json.loads(line) for line in forked.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The source entries are copied with their ids, then the new prompt continues the fork.
    source_ids = {record["id"] for record in source if "id" in record}
    fork_ids = {record["id"] for record in records if "id" in record}
    assert source_ids <= fork_ids
    assert len(records) > len(source)


def test_fork_rejects_existing_target_and_bad_source(agent_env):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "exists", "one") == 0
    assert run_cli("--provider", "faux", "--no-mcp", "--fork", "exists",
                   "--session-id", "exists", "two") == 1
    assert run_cli("--provider", "faux", "--no-mcp", "--fork", "missing-id", "two") == 1


def test_resume_requires_a_terminal(agent_env, capsys):
    assert run_cli("--provider", "faux", "--no-mcp", "--resume") == 2
    assert "--resume requires an interactive terminal" in capsys.readouterr().err


def test_resume_picks_session_from_list(agent_env, monkeypatch):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "pickme", "one") == 0
    target = agent_env / ".pi" / "sessions" / "pickme.jsonl"
    before = len(target.read_text(encoding="utf-8").splitlines())

    choices = {}

    async def fake_choose(title, values, **kwargs):
        choices[title] = values
        return str(target)

    monkeypatch.setattr(cli, "_choose_console", fake_choose)

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(cli.sys, "stdin", FakeTty())
    assert run_cli("--provider", "faux", "--no-mcp", "--resume") == 0
    assert str(target) in choices["Resume session"][0][1]
    assert len(target.read_text(encoding="utf-8").splitlines()) > before


# -- export -----------------------------------------------------------------

TEMPLATE_PLACEHOLDERS = ("{{CSS}}", "{{JS}}", "{{SESSION_DATA}}", "{{MARKED_JS}}",
                         "{{HIGHLIGHT_JS}}", "{{THEME_VARS}}", "{{BODY_BG}}",
                         "{{CONTAINER_BG}}", "{{INFO_BG}}")


def test_export_cli_writes_single_file_html(agent_env):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "exported", "hello") == 0
    source = agent_env / ".pi" / "sessions" / "exported.jsonl"
    out = agent_env / "out.html"
    assert run_cli("--no-mcp", "--export", str(source), str(out)) == 0
    html = out.read_text(encoding="utf-8")
    for placeholder in TEMPLATE_PLACEHOLDERS:
        assert placeholder not in html
    assert html.count("<script>") == 3  # marked, highlight.js, application code
    assert 'src="http' not in html  # no external references: one self-contained file
    match = re.search(r'id="session-data" type="application/json">([^<]+)<', html)
    data = json.loads(base64.b64decode(match.group(1)))
    assert data["header"]["type"] == "session"
    assert any(entry["type"] == "message" for entry in data["entries"])
    assert data["leafId"]


def test_export_cli_jsonl_uses_ts_format(agent_env):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "tsout", "hello") == 0
    source = agent_env / ".pi" / "sessions" / "tsout.jsonl"
    out = agent_env / "out-ts.jsonl"
    assert run_cli("--no-mcp", "--export", str(source), str(out)) == 0
    header = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert header["type"] == "session" and header["version"] == 3


def test_export_cli_errors(agent_env, capsys):
    assert run_cli("--no-mcp", "--export", str(agent_env / "missing.jsonl")) == 1
    assert "File not found" in capsys.readouterr().err


# -- print ------------------------------------------------------------------

def test_print_flag_runs_one_shot_and_reads_stdin(agent_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped prompt\n"))
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--print") == 0
    assert "Faux response." in capsys.readouterr().out


def test_print_flag_without_prompt_fails(agent_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--print") == 2
    assert "--print requires a prompt" in capsys.readouterr().err


# -- runtime switches -------------------------------------------------------

def test_offline_and_verbose_flags(agent_env, monkeypatch, capsys):
    monkeypatch.setenv("PI_OFFLINE", "0")
    monkeypatch.setenv("PI_TIMING", "0")
    monkeypatch.setattr(cli_timings, "ENABLED", False)
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--offline", "--verbose", "hi") == 0
    assert os.environ["PI_OFFLINE"] == "1"
    assert "Startup Timings" in capsys.readouterr().err


def test_use_theme_runtime_override(agent_env):
    theme_dir = agent_env / "agent" / "themes"
    theme_dir.mkdir(parents=True)
    (theme_dir / "custom.json").write_text(json.dumps({"prompt": "bold"}), encoding="utf-8")
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--use-theme", "custom", "hi") == 0
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--use-theme", "missing", "hi") == 2


def test_theme_flag_loads_themes_outside_agent_dir(agent_env, tmp_path):
    external = tmp_path / "more-themes"
    external.mkdir()
    (external / "extra.json").write_text(json.dumps({"prompt": "bold"}), encoding="utf-8")
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session",
                   "--theme", str(external), "--use-theme", "extra", "hi") == 0
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--theme", str(external / "nope.json"), "hi") == 2


# -- resource switches ------------------------------------------------------

SKILL_MD = """---
name: demo-skill
description: A demo skill for tests
---

Skill body.
"""


def _skill_system_section(agent_env, records) -> str:
    return system_record(records)["message"].get("sections", {}).get("skills") or ""


def test_skill_flags(agent_env):
    project_skills = agent_env / ".pi" / "skills" / "demo"
    project_skills.mkdir(parents=True)
    (project_skills / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    explicit = agent_env / "extra-skill"
    explicit.mkdir()
    (explicit / "SKILL.md").write_text(SKILL_MD.replace("demo-skill", "extra-skill"), encoding="utf-8")

    assert run_cli("--provider", "faux", "--no-mcp", "--approve", "hi") == 0
    assert "demo-skill" in _skill_system_section(agent_env, latest_session(agent_env)["records"])
    assert run_cli("--provider", "faux", "--no-mcp", "--approve", "--no-skills", "hi") == 0
    assert _skill_system_section(agent_env, latest_session(agent_env)["records"]) == ""
    assert run_cli("--provider", "faux", "--no-mcp", "--approve",
                   "--no-skills", "--skill", str(explicit), "hi") == 0
    assert "extra-skill" in _skill_system_section(agent_env, latest_session(agent_env)["records"])
    assert "demo-skill" not in _skill_system_section(agent_env, latest_session(agent_env)["records"])


def test_extension_flags(agent_env, monkeypatch):
    project_ext = agent_env / ".pi" / "extensions" / "marker.py"
    project_ext.parent.mkdir(parents=True)
    marker = agent_env / "marker.log"
    project_ext.write_text(
        "import os\n"
        "def setup(api):\n"
        "    with open(os.environ['EXT_MARKER'], 'a', encoding='utf-8') as handle:\n"
        "        handle.write('loaded\\n')\n", encoding="utf-8")
    monkeypatch.setenv("EXT_MARKER", str(marker))

    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--approve", "hi") == 0
    assert marker.read_text(encoding="utf-8") == "loaded\n"
    marker.unlink()
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--approve",
                   "--no-extensions", "hi") == 0
    assert not marker.exists()
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--approve",
                   "--no-extensions", "-e", str(project_ext), "hi") == 0
    assert marker.read_text(encoding="utf-8") == "loaded\n"
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session",
                   "-e", str(agent_env / "missing-ext.py"), "hi") == 2


# -- models scope -----------------------------------------------------------

def test_models_scope_limits_registry_cycle(agent_env):
    registry = ModelRegistry(Configuration(str(agent_env), str(agent_env / "agent")))
    registry.model_scope = ["faux/*"]
    assert [model.provider for model in registry.scoped_models()] == ["faux"]
    registry.model_scope = ["openai/gpt-4o-mini"]
    assert [model.id for model in registry.scoped_models()] == ["gpt-4o-mini"]
    registry.model_scope = ["no-such-provider/*"]
    assert registry.scoped_models() == []
    registry.model_scope = None
    assert registry.scoped_models() == list(registry.models.values())


def test_models_scope_rejects_empty_patterns(agent_env, capsys):
    assert run_cli("--provider", "faux", "--no-mcp", "--no-session", "--offline",
                   "--models", ",,,", "hi") == 2
    assert "--models requires at least one provider/model pattern" in capsys.readouterr().err


# -- session internals ------------------------------------------------------

def test_start_new_with_session_id(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "s"), cwd=str(tmp_path))
    assert manager.start_new("abc_123").endswith("abc_123.jsonl")
    with pytest.raises(ValueError):
        manager.start_new("-bad-")
    with pytest.raises(ValueError):
        manager.start_new("abc_123")  # already in use


def test_fork_from_copies_tree_and_name(tmp_path):
    source = SessionManager(session_dir=str(tmp_path / "s"), cwd=str(tmp_path))
    source.start_new()
    source.rename("origin")
    source.append(UserMessage(content="root", timestamp=0))
    source.append(UserMessage(content="child", timestamp=1))
    target = SessionManager(session_dir=str(tmp_path / "t"), cwd=str(tmp_path))
    messages = target.fork_from(source.session_file)
    assert [message.content for message in messages] == ["root", "child"]
    assert target.name == "origin"
    assert target.leaf_id == source.leaf_id
    assert {entry.id for entry in target.entries} == {entry.id for entry in source.entries}
    reloaded = SessionManager(session_dir=str(tmp_path / "t"), cwd=str(tmp_path))
    reloaded.load(target.session_file)
    assert [message.content for message in reloaded.messages()] == ["root", "child"]
    assert reloaded.name == "origin"


def test_export_session_file_default_output(agent_env, monkeypatch):
    assert run_cli("--provider", "faux", "--no-mcp", "--session-id", "defexp", "hello") == 0
    source = agent_env / ".pi" / "sessions" / "defexp.jsonl"
    monkeypatch.chdir(agent_env)
    out = export_session_file(str(source))
    assert out == str(agent_env / f"pi-session-defexp.html")
    assert os.path.exists(out)


# -- template unit ----------------------------------------------------------

def test_generate_html_replaces_all_placeholders():
    html = generate_html({"header": {"type": "session", "version": 3}, "entries": [], "leafId": None})
    for placeholder in TEMPLATE_PLACEHOLDERS:
        assert placeholder not in html
    assert "marked" in html.lower() and "hljs" in html.lower()
