import asyncio
from pathlib import Path

import pytest

from pi.ai.models import resolve_model
from pi.ai.providers.faux import reset_script_index
from pi.agent.agent import Agent
from pi.coding_agent.cli import CliRenderer, async_main, build_parser, interactive_loop, make_agent
from pi.coding_agent.config import Configuration
from pi.coding_agent.resources import ResourceCatalog, parse_arguments, substitute_arguments
from pi.coding_agent.session import SessionManager
from pi.coding_agent.model_registry import ModelRegistry
from pi.ai.types import TranscriptContext
from pi.ai.utils import normalize_context


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def resources(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / ".git").mkdir()
    config = Configuration(str(cwd), str(home / ".pi-python"))
    return config


def test_skill_discovery_metadata_only_and_explicit_loading(resources):
    write(resources.cwd / ".pi/skills/pdf/SKILL.md", "---\nname: pdf\ndescription: >\n  Read PDF files\n  and extract text\n---\nSECRET FULL INSTRUCTIONS")
    write(resources.agent_dir / "skills/manual/SKILL.md", "---\nname: manual\ndescription: Manual only\ndisable-model-invocation: true\n---\nExplicit instructions")
    catalog = ResourceCatalog(resources)
    assert set(catalog.skills) == {"pdf", "manual"}
    prompt = catalog.system_prompt()
    assert "Read PDF files" in prompt and "SKILL.md" in prompt
    assert "SECRET FULL INSTRUCTIONS" not in prompt
    assert "manual" not in prompt
    assert "SECRET FULL INSTRUCTIONS" in catalog.expand("/skill:pdf extract report.pdf")
    assert "extract report.pdf" in catalog.expand("/skill:pdf extract report.pdf")
    assert "Explicit instructions" in catalog.expand("/skill:manual")
    with pytest.raises(ValueError, match="Unknown skill"):
        catalog.expand("/skill:missing")


def test_skill_duplicates_invalid_and_ancestor_boundary(resources):
    write(resources.cwd / ".pi/skills/a/SKILL.md", "---\nname: same\ndescription: Project\n---\nA")
    write(resources.agent_dir / "skills/b/SKILL.md", "---\nname: same\ndescription: User\n---\nB")
    write(resources.cwd / ".pi/skills/invalid/SKILL.md", "---\nname: BAD\ndescription: invalid\n---\nBad")
    write(resources.cwd / ".agents/skills/local/SKILL.md", "---\nname: local\ndescription: Local\n---\nLocal")
    write(resources.cwd.parent / ".agents/skills/outside/SKILL.md", "---\nname: outside\ndescription: Outside\n---\nOutside")
    catalog = ResourceCatalog(resources)
    assert catalog.skills["same"].description == "Project"
    assert "local" in catalog.skills and "outside" not in catalog.skills
    assert len(catalog.diagnostics) == 2


def test_templates_discovery_reload_and_reserved_names(resources):
    write(resources.cwd / ".pi/prompts/review.md", '---\ndescription: Review code\nargument-hint: "[focus]"\n---\nReview ${1:-correctness}; $@')
    write(resources.cwd / ".pi/prompts/nested/hidden.md", "Not discovered")
    write(resources.agent_dir / "prompts/help.md", "Must not replace help")
    write(resources.agent_dir / "prompts/broken.md", "---\ndescription: [invalid\n---\nBody")
    catalog = ResourceCatalog(resources)
    assert set(catalog.prompts) == {"review"}
    assert catalog.prompts["review"].argument_hint == "[focus]"
    assert catalog.expand('/review "API compatibility" tests') == "Review API compatibility; API compatibility tests"
    assert catalog.expand("/review") == "Review correctness; "
    assert catalog.expand("ordinary prompt") == "ordinary prompt"
    assert catalog.expand("/unknown") == "/unknown"
    assert len(catalog.diagnostics) == 2
    write(resources.cwd / ".pi/prompts/review.md", "Changed $1")
    catalog.reload()
    assert catalog.expand("/review value") == "Changed value"


def test_template_argument_forms_and_no_recursive_expansion():
    assert parse_arguments('"API compatibility" \'one two\' C:\\work\\pi ""') == ["API compatibility", "one two", "C:\\work\\pi", ""]
    assert substitute_arguments("$1|$2|$@|$ARGUMENTS|${@:2}|${@:2:1}", ["a", "b", "c"]) == "a|b|a b c|a b c|b c|b"
    assert substitute_arguments("${1:-fallback}|${@:-all}|${ARGUMENTS:-other}", []) == "fallback|all|other"
    assert substitute_arguments("$1 ${2:-$@}", ["$2"]) == "$2 $@"
    assert substitute_arguments("${@:0:0}|$0", ["x"]) == "|"
    with pytest.raises(ValueError):
        parse_arguments('"unterminated')


def test_explicit_paths_and_disable_defaults(resources):
    write(resources.cwd / ".pi/extra/special.md", "Explicit $1")
    write(resources.cwd / ".pi/prompts/default.md", "Default")
    resources.save("prompts", ["extra/special.md"], project=True)
    resources.save("include_default_resources", False)
    catalog = ResourceCatalog(resources)
    assert set(catalog.prompts) == {"special"}
    assert catalog.expand("/special argument") == "Explicit argument"
    with pytest.raises(ValueError):
        resources.save("skills", "not a list")


def test_cli_one_shot_template_is_expanded_and_persisted(resources, monkeypatch):
    monkeypatch.chdir(resources.cwd)
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(resources.agent_dir))
    write(resources.cwd / ".pi/prompts/review.md", "Review $1")
    reset_script_index()
    # The project .pi resources require trust; one-shot mode passes --approve.
    assert asyncio.run(async_main(["--provider", "faux", "--approve", "/review", "tests"])) == 0
    session = SessionManager(cwd=str(resources.cwd))
    messages = session.load(session.latest_session_file())
    assert any(message.role == "user" and message.content == "Review tests" for message in messages)


def test_interactive_skill_and_template_commands(resources, monkeypatch):
    write(resources.cwd / ".pi/skills/demo/SKILL.md", "---\nname: demo\ndescription: Demo skill\n---\nSkill instructions")
    write(resources.cwd / ".pi/prompts/review.md", "Review $1")
    catalog = ResourceCatalog(resources)
    inputs = iter(["/skill:demo task", "/review tests", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setenv("FAUX_SCRIPT", '[{"text":"one"},{"text":"two"}]')
    reset_script_index()
    agent = Agent(resolve_model("faux"))
    asyncio.run(interactive_loop(agent, CliRenderer(quiet=True), None, resources=catalog))
    prompts = [message.content for message in agent.messages if message.role == "user"]
    assert "Skill instructions" in prompts[0] and "task" in prompts[0]
    assert prompts[1] == "Review tests"


def test_reload_updates_skill_system_state(resources, monkeypatch):
    path = write(resources.cwd / ".pi/skills/demo/SKILL.md", "---\nname: demo\ndescription: Old description\n---\nBody")
    catalog = ResourceCatalog(resources)
    registry = ModelRegistry(resources)
    agent = make_agent(build_parser().parse_args(["--provider", "faux"]), str(resources.cwd), registry, catalog)
    session = SessionManager(cwd=str(resources.cwd))
    for message in agent.messages:
        session.append(message)
    commands = iter(["/reload", "/exit"])

    def input_command(prompt):
        write(path, "---\nname: demo\ndescription: New description\n---\nNew instructions")
        return next(commands)

    monkeypatch.setattr("builtins.input", input_command)
    asyncio.run(interactive_loop(agent, CliRenderer(quiet=True), session, registry, catalog))
    system, _ = normalize_context(TranscriptContext(agent.messages))
    assert "New description" in system.sections["skills"]
    assert "Old description" not in system.sections["skills"]
    loaded = session.load(session.session_file)
    assert loaded[-1].sections["skills"] == system.sections["skills"]


def test_resume_refreshes_skill_catalog(resources, monkeypatch):
    monkeypatch.chdir(resources.cwd)
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(resources.agent_dir))
    path = write(resources.cwd / ".pi/skills/demo/SKILL.md", "---\nname: demo\ndescription: Before\n---\nBody")
    reset_script_index()
    asyncio.run(async_main(["--provider", "faux", "--approve", "hello"]))
    write(path, "---\nname: demo\ndescription: After\n---\nBody")
    reset_script_index()
    asyncio.run(async_main(["--provider", "faux", "--approve", "--continue", "again"]))
    session = SessionManager(cwd=str(resources.cwd))
    system, _ = normalize_context(TranscriptContext(session.load(session.latest_session_file())))
    assert "After" in system.sections["skills"]
    assert "Before" not in system.sections["skills"]


# -- v1.0.4 port: the skills hint follows the declared file reader (c30840c2e)


def test_skills_prompt_names_the_declared_file_reader(resources):
    write(resources.cwd / ".pi/skills/demo/SKILL.md", "---\nname: demo\ndescription: Demo skill\n---\nBody")
    catalog = ResourceCatalog(resources)
    read_prompt = catalog.system_prompt(file_read_tool="read")
    assert "use read to load its SKILL.md" in read_prompt
    assert "use bash to load its SKILL.md" in catalog.system_prompt(file_read_tool="bash")
    indirect = catalog.system_prompt(file_read_tool="indirect")
    assert "load its SKILL.md" in indirect
    assert "use read" not in indirect and "use bash" not in indirect
    assert catalog.system_prompt(file_read_tool=None) == ""


def test_make_agent_tools_flag_shapes_tools_and_system_prompt(resources):
    registry = ModelRegistry(resources)
    agent = make_agent(build_parser().parse_args(["--provider", "faux", "--tools", "read,grep"]),
                       str(resources.cwd), registry)
    assert [tool.name for tool in agent.tools] == ["read", "grep"]
    system = agent.messages[0].content
    assert "- read: Read file contents" in system
    assert "- bash:" not in system


def test_make_agent_no_tools_leaves_no_tool_section(resources):
    registry = ModelRegistry(resources)
    agent = make_agent(build_parser().parse_args(["--provider", "faux", "--no-tools"]),
                       str(resources.cwd), registry)
    assert agent.tools == []
    assert "(none)" in agent.messages[0].content


def test_make_agent_exclude_tools_flag(resources):
    registry = ModelRegistry(resources)
    agent = make_agent(build_parser().parse_args(["--provider", "faux", "--exclude-tools", "bash,edit"]),
                       str(resources.cwd), registry)
    names = {tool.name for tool in agent.tools}
    assert "bash" not in names and "edit" not in names
    assert "read" in names and "write" in names
