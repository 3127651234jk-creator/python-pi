"""Port of packages/evals/test/harness.test.ts, plus a faux-provider run."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from pi.coding_agent.system_prompt import (
    SystemPromptOptions,
    build_system_prompt,
    get_docs_path,
    get_examples_path,
    get_readme_path,
)
from pi.evals.framework import HarnessRunArgs
from pi.evals.report import PI_SESSION_SNAPSHOT_ARTIFACT
from pi.evals.harness import (
    DOCUMENTATION_EVAL_TOOLS,
    ModelSelection,
    PiCodingAgentHarnessOptions,
    apply_isolated_environment,
    create_pi_documentation_eval_harness,
    exclude_pi_documentation,
    resolve_documentation_variant,
    resolve_model_selection,
    run_pi_coding_agent,
    verify_system_prompt,
)


def test_resolve_model_selection_prefers_an_explicit_harness_model():
    selection = resolve_model_selection(
        ModelSelection(provider="anthropic", id="claude-opus-4-6"),
        {"PI_PROVIDER": "openai-codex", "PI_MODEL": "gpt-5.6-sol"},
    )
    assert (selection.provider, selection.id) == ("anthropic", "claude-opus-4-6")


def test_resolve_model_selection_uses_trimmed_environment_defaults():
    selection = resolve_model_selection(
        None, {"PI_PROVIDER": " openai-codex ", "PI_MODEL": " gpt-5.6-sol "})
    assert (selection.provider, selection.id) == ("openai-codex", "gpt-5.6-sol")


@pytest.mark.parametrize("environment", [{}, {"PI_PROVIDER": "openai-codex"}, {"PI_MODEL": "gpt-5.6-sol"}])
def test_resolve_model_selection_rejects_incomplete_selection(environment):
    with pytest.raises(ValueError, match="Select a harness model explicitly"):
        resolve_model_selection(None, environment)


def test_isolate_process_environment_removes_runner_metadata_and_restores(monkeypatch):
    monkeypatch.setenv("PI_EVAL_VARIANT", "with_docs")
    monkeypatch.setenv("PI_EVAL_ARTIFACT_DIR", "/tmp/artifacts")
    old_home = os.environ.get("HOME")
    old_profile = os.environ.get("USERPROFILE")
    old_agent_dir = os.environ.get("PI_PY_AGENT_DIR")
    restore = apply_isolated_environment("/tmp/eval-home", "/tmp/eval-agent")
    try:
        assert os.environ["HOME"] == "/tmp/eval-home"
        assert os.environ["USERPROFILE"] == "/tmp/eval-home"
        assert os.environ["PI_PY_AGENT_DIR"] == "/tmp/eval-agent"
        assert "PI_EVAL_VARIANT" not in os.environ
        assert "PI_EVAL_ARTIFACT_DIR" not in os.environ
    finally:
        restore()
    assert os.environ.get("HOME") == old_home
    assert os.environ.get("USERPROFILE") == old_profile
    assert os.environ.get("PI_PY_AGENT_DIR") == old_agent_dir
    assert os.environ["PI_EVAL_VARIANT"] == "with_docs"
    assert os.environ["PI_EVAL_ARTIFACT_DIR"] == "/tmp/artifacts"


@pytest.mark.parametrize("variant", ["without_docs", "with_docs"])
def test_accepts_documentation_variants(monkeypatch, variant):
    assert resolve_documentation_variant(variant) == variant


@pytest.mark.parametrize("variant", [None, "", "other"])
def test_rejects_invalid_documentation_variants(monkeypatch, variant):
    monkeypatch.delenv("PI_EVAL_VARIANT", raising=False)
    with pytest.raises(TypeError, match="PI_EVAL_VARIANT"):
        resolve_documentation_variant(variant)


def build_eval_prompt():
    return build_system_prompt(SystemPromptOptions(
        cwd="/workspace",
        selected_tools=list(DOCUMENTATION_EVAL_TOOLS),
        include_pi_documentation=True,
    ))


def test_strips_only_the_documentation_routing_section_from_the_default_prompt():
    prompt = build_eval_prompt()
    assert "\n<docs>\nPi documentation (read only" in prompt
    assert "\n<rules>\n" in prompt
    assert "\n<cwd>\n/workspace\n</cwd>" in prompt
    assert get_readme_path() in prompt

    stripped = exclude_pi_documentation(prompt)
    assert "\n<rules>\n" in stripped
    assert "\n<cwd>\n/workspace\n</cwd>" in stripped
    assert "<docs>" not in stripped
    assert "Pi documentation" not in stripped
    assert get_readme_path() not in stripped
    assert get_docs_path() not in stripped
    assert get_examples_path() not in stripped


def test_verifies_the_prompt_that_was_sent():
    prompt = build_eval_prompt()
    stripped = exclude_pi_documentation(prompt)

    assert verify_system_prompt(stripped, name="without_docs", expected_pi_documentation=False) == stripped
    with pytest.raises(RuntimeError, match="does not match"):
        verify_system_prompt(prompt, name="without_docs", expected_pi_documentation=False)


def test_fails_closed_when_prompt_markers_are_missing():
    with pytest.raises(RuntimeError, match="no Pi documentation section"):
        exclude_pi_documentation("Instructions")
    with pytest.raises(RuntimeError, match="no working-directory section"):
        exclude_pi_documentation("\n<docs>\nPi documentation\n</docs>")


def test_rejects_documentation_harnesses_outside_the_container_sandbox(monkeypatch):
    monkeypatch.delenv("PI_EVAL_CONTAINER", raising=False)
    monkeypatch.delenv("PI_EVAL_SANDBOX_UID", raising=False)
    monkeypatch.delenv("PI_EVAL_SANDBOX_GID", raising=False)
    with pytest.raises(RuntimeError, match="isolated container sandbox"):
        create_pi_documentation_eval_harness()


def test_run_pi_coding_agent_completes_against_the_faux_provider(tmp_path, monkeypatch):
    from pi.ai.providers.faux import reset_script_index

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([{"text": "Faux eval answer."}]))
    reset_script_index()
    artifacts = {}

    args = HarnessRunArgs(
        input="Say the answer.",
        artifacts=artifacts,
        set_artifact=lambda name, value: artifacts.__setitem__(name, value),
    )
    options = PiCodingAgentHarnessOptions(name="faux-eval", model=ModelSelection(provider="faux", id="faux"))
    result = asyncio.run(run_pi_coding_agent(args, options))

    assert result["output"] == "Faux eval answer."
    assert artifacts["runId"]
    events = result["events"]
    assert {"type": "message", "role": "user", "content": "Say the answer."} in events
    assert {"type": "message", "role": "assistant", "content": "Faux eval answer."} in events
    assert result["usage"]["provider"] == "faux"
    assert result["usage"]["model"] == "faux"
    assert result["usage"]["totalTokens"] > 0
    assert "systemPromptSha256" in result["metadata"]
    session_lines = artifacts[PI_SESSION_SNAPSHOT_ARTIFACT].strip().splitlines()
    assert session_lines
    assert json.loads(session_lines[0])["type"] == "message"


def test_run_pi_coding_agent_preserves_partial_diagnostics_on_failure(monkeypatch):
    from pi.ai.providers.faux import reset_script_index

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([{"text": "Faux eval answer."}]))
    reset_script_index()
    artifacts = {}

    def exploding_output(**_kwargs):
        raise RuntimeError("grader exploded")

    args = HarnessRunArgs(
        input="Say the answer.",
        artifacts=artifacts,
        set_artifact=lambda name, value: artifacts.__setitem__(name, value),
    )
    options = PiCodingAgentHarnessOptions(name="faux-eval", model=ModelSelection(provider="faux", id="faux"),
                                          output=exploding_output)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(run_pi_coding_agent(args, options))
    assert getattr(excinfo.value, "harness_run", None) is not None
    partial = excinfo.value.harness_run
    assert partial["errors"]
    assert partial["session"]["events"]
    assert "totalMs" in partial["timings"]


ACME_MODEL = {
    "id": "acme-chat",
    "name": "Acme Chat",
    "provider": "acme",
    "reasoning": False,
    "input": ["text"],
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 32768,
    "maxTokens": 4096,
}


def test_reload_step_reloads_the_registry_for_the_output_hook(tmp_path, monkeypatch, direct_httpx):
    """Prompt, reload, then probe the reloaded registry (openai-provider eval flow)."""
    from evals.acme_server import (
        OPENAI_MODEL_ID,
        OPENAI_PROBE_PROMPT,
        OPENAI_PROBE_RESPONSE,
        OPENAI_PROVIDER_ID,
        create_acme_server,
    )
    from evals.configured_runtime import ProviderScenario, inspect_provider
    from pi.agent.types import AgentTool, AgentToolResult
    from pi.ai.providers.base import SimpleStreamOptions
    from pi.ai.providers.faux import reset_script_index
    from pi.ai.types import TextContent, TranscriptContext, UserMessage
    from pi.coding_agent.config import write_json

    monkeypatch.setenv("ACME_API_KEY", "resolved-acme-key")
    server = create_acme_server("openai")
    server.start()
    try:
        def configure_acme_tool():
            async def execute(tool_call_id, params, signal=None, on_update=None):
                agent_dir = Path(os.environ["PI_PY_AGENT_DIR"])
                write_json(agent_dir / "models.json", {"providers": {
                    OPENAI_PROVIDER_ID: {
                        "base_url": server.base_url(),
                        "api": "openai-completions",
                        "api_key": "$ACME_API_KEY",
                        "models": [{
                            "id": ACME_MODEL["id"],
                            "name": ACME_MODEL["name"],
                            "reasoning": ACME_MODEL["reasoning"],
                            "input": ACME_MODEL["input"],
                            "costs": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                            "context_window": ACME_MODEL["contextWindow"],
                            "max_tokens": ACME_MODEL["maxTokens"],
                        }],
                    },
                }})
                return AgentToolResult(content=[TextContent(text="Provider configured.")])

            return AgentTool(
                name="configure_acme",
                label="Configure acme",
                description="Write the acme provider into the agent models.json.",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )

        async def probe_output(**kwargs):
            return await inspect_provider(kwargs["registry"], ProviderScenario(
                provider_id=OPENAI_PROVIDER_ID,
                model_id=OPENAI_MODEL_ID,
                create_context=lambda: TranscriptContext(
                    [UserMessage(content=OPENAI_PROBE_PROMPT, timestamp=0)]),
                options=SimpleStreamOptions(max_tokens=32),
                valid_request_received=server.valid_request_received,
            ))

        monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
            {"toolCalls": [{"name": "configure_acme", "arguments": {}}]},
            {"text": "Provider configured."},
        ]))
        reset_script_index()
        artifacts = {}
        args = HarnessRunArgs(
            input=[{"type": "prompt", "content": "Configure the acme provider."},
                   {"type": "reload"}],
            artifacts=artifacts,
            set_artifact=lambda name, value: artifacts.__setitem__(name, value),
        )
        options = PiCodingAgentHarnessOptions(
            name="reload-eval",
            model=ModelSelection(provider="faux", id="faux"),
            custom_tools=[configure_acme_tool()],
            output=probe_output,
        )
        result = asyncio.run(run_pi_coding_agent(args, options))

        assert result["output"] == {"result": {
            "validRequestReceived": True,
            "model": ACME_MODEL,
            "response": {"text": OPENAI_PROBE_RESPONSE, "stopReason": "stop",
                         "inputTokens": 3, "outputTokens": 2},
        }}
        assert server.valid_request_received() is True
    finally:
        server.stop()


def test_reload_step_reports_a_broken_configuration_to_the_output_hook(monkeypatch, direct_httpx):
    """A models.json the agent broke surfaces as a structured probe error."""
    from evals.configured_runtime import ProviderScenario, inspect_provider
    from pi.agent.types import AgentTool, AgentToolResult
    from pi.ai.providers.faux import reset_script_index
    from pi.ai.types import TextContent, TranscriptContext, UserMessage

    def break_models_tool():
        async def execute(tool_call_id, params, signal=None, on_update=None):
            agent_dir = Path(os.environ["PI_PY_AGENT_DIR"])
            (agent_dir / "models.json").write_text("{", encoding="utf-8")
            return AgentToolResult(content=[TextContent(text="Configuration written.")])

        return AgentTool(
            name="break_models",
            label="Break models",
            description="Write an invalid models.json.",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )

    async def probe_output(**kwargs):
        return await inspect_provider(kwargs["registry"], ProviderScenario(
            provider_id="acme", model_id="acme-chat",
            create_context=lambda: TranscriptContext([UserMessage(content="hi", timestamp=0)]),
        ))

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"name": "break_models", "arguments": {}}]},
        {"text": "Configuration written."},
    ]))
    reset_script_index()
    args = HarnessRunArgs(input=[{"type": "prompt", "content": "Write configuration."},
                                 {"type": "reload"}])
    options = PiCodingAgentHarnessOptions(
        name="reload-eval",
        model=ModelSelection(provider="faux", id="faux"),
        custom_tools=[break_models_tool()],
        output=probe_output,
    )
    result = asyncio.run(run_pi_coding_agent(args, options))
    assert "Cannot read configuration" in result["output"]["result"]["error"]


def test_unknown_input_step_types_are_rejected():
    args = HarnessRunArgs(input=[{"type": "restart"}])
    options = PiCodingAgentHarnessOptions(model=ModelSelection(provider="faux", id="faux"))
    with pytest.raises(ValueError, match="Unknown eval input step type"):
        asyncio.run(run_pi_coding_agent(args, options))
