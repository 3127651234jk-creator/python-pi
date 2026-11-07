"""Single-page custom API form: layout, focus, validation and save flow
(docs/specs/custom-api-single-page.md). Offline only: fake terminal sizes,
pipe input and a mock save service."""

import asyncio
from pathlib import Path

import httpx
import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.cli import custom_login_command
from pi.coding_agent.config import Configuration, read_json, write_json
from pi.coding_agent.custom_api import (
    ConcurrentConfigChange, CustomApiForm, validate_form_values,
)
from pi.coding_agent.custom_api_form import (
    CustomApiFormApp, CustomApiFormResult, FormDraft, SaveResult,
)
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.session import SessionManager


class FixedOutput(DummyOutput):
    """DummyOutput with a size the test controls."""

    def __init__(self, columns=100, rows=36):
        self.size = Size(rows, columns)

    def get_size(self):
        return self.size


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    for key in ("PI_PY_PROVIDER", "PI_PY_MODEL", "PI_PY_API_KEY", "PI_PY_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return Configuration(str(tmp_path))


def draft(**overrides):
    values = dict(
        provider_id="my-gateway",
        base_url="https://gateway.example.test/v1",
        api_key="secret-key",
        model_id="vendor/model-a",
    )
    values.update(overrides)
    return FormDraft(**values)


class SaveService:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = list(outcomes or [])

    async def __call__(self, form, use_now, set_default):
        self.calls.append((form, use_now, set_default))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return SaveResult([f"saved {form.provider_id}/{form.model_id}"])


async def start(app):
    task = asyncio.create_task(app.run())
    for _ in range(200):
        if app.app.is_running and app.app.layout.visible_windows:
            await asyncio.sleep(0.1)
            return task
        await asyncio.sleep(0.01)
    raise AssertionError("form did not start")


async def send(pipe, text):
    pipe.send_text(text)
    await asyncio.sleep(0.05)


async def finish(task):
    return await asyncio.wait_for(task, 2)


def make_app(pipe, config, *, initial=None, save=None, shortcuts=None, width=100, height=36):
    output = FixedOutput(width, height)
    app = CustomApiFormApp(
        validate=lambda values: validate_form_values(values, {}, config),
        save=save or SaveService(),
        style=Style.from_dict({}),
        shortcuts=shortcuts,
        initial=initial,
        input=pipe,
        output=output,
    )
    return app, output


async def with_pipe(body):
    # Win32PipeInput binds to the running loop, so create it inside asyncio.
    with create_pipe_input() as pipe:
        await body(pipe)


def test_form_starts_on_provider_id_and_tab_order(config):
    async def body(pipe):
        app, _output = make_app(pipe, config, initial=draft())
        task = await start(app)
        assert app.app.layout.has_focus(app.inputs["provider_id"].control)
        await send(pipe, "\t")
        assert app.app.layout.has_focus(app.inputs["base_url"].control)
        await send(pipe, "\x1b[Z")
        assert app.app.layout.has_focus(app.inputs["provider_id"].control)
        await send(pipe, "\x1b[Z")
        assert app.app.layout.has_focus(app.api_selector.control)
        await send(pipe, "\t")
        assert app.app.layout.has_focus(app.inputs["provider_id"].control)
        await send(pipe, "\x1b")
        assert await finish(task) is None

    asyncio.run(with_pipe(body))


def test_enter_moves_to_the_next_field(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\r")
        assert app.app.layout.has_focus(app.inputs["base_url"].control)
        assert save.calls == []
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_space_toggles_a_checkbox_and_enter_presses_a_button(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        for _ in range(7):
            await send(pipe, "\t")
        assert app.app.layout.has_focus(app.checkboxes["image_input"].control)
        assert not app.checkboxes["image_input"].checked
        await send(pipe, " ")
        assert app.checkboxes["image_input"].checked
        await send(pipe, " ")
        assert not app.checkboxes["image_input"].checked
        for _ in range(5):  # reasoning, use now, default, cancel, save
            await send(pipe, "\t")
        assert app.app.layout.has_focus(app.save_button.control)
        await send(pipe, "\r")
        result = await finish(task)
        assert result is not None and len(save.calls) == 1

    asyncio.run(with_pipe(body))


def test_save_valid_form_calls_the_service_once(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        result = await finish(task)
        assert isinstance(result, CustomApiFormResult)
        assert len(save.calls) == 1
        form, use_now, set_default = save.calls[0]
        assert form.provider_id == "my-gateway" and form.model_id == "vendor/model-a"
        assert form.api_key == "secret-key"
        assert form.display_name == "vendor/model-a"
        assert use_now is True and set_default is False
        assert result.messages == ["saved my-gateway/vendor/model-a"]

    asyncio.run(with_pipe(body))


def test_save_shows_all_errors_and_focuses_the_first(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(
            pipe, config, initial=draft(provider_id="", base_url="", context_window="100",
                                  max_tokens="200"), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        assert save.calls == []
        assert app.app.is_running
        assert set(app.draft.errors) >= {"provider_id", "base_url"}
        assert app.app.layout.has_focus(app.inputs["provider_id"].control)
        assert "secret-key" not in repr(app.draft)
        assert "secret-key" not in " ".join(app.draft.errors.values())
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_save_failure_stays_in_the_form_and_keeps_the_key(config):
    async def body(pipe):
        save = SaveService([OSError("disk full")])
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        assert app.app.is_running
        assert "disk full" in app._status
        assert app.inputs["api_key"].text == "secret-key"
        assert "secret-key" not in app._status
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None and len(save.calls) == 2

    asyncio.run(with_pipe(body))


def test_concurrent_change_keeps_the_draft_and_asks_again(config):
    async def body(pipe):
        save = SaveService([ConcurrentConfigChange("changed"), SaveResult(["ok"])])
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        assert app.app.is_running
        assert "changed on disk" in app._status
        assert app.inputs["api_key"].text == "secret-key"
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None and len(save.calls) == 2
        assert result.form.provider_id == "my-gateway"

    asyncio.run(with_pipe(body))


def test_cancel_destroys_the_draft_and_never_saves(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x1b")
        assert await finish(task) is None
        assert save.calls == []
        assert app.draft.api_key == ""
        assert app.inputs["api_key"].text == ""

    asyncio.run(with_pipe(body))


def test_ctrl_c_also_cancels(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x03")
        assert await finish(task) is None
        assert save.calls == []

    asyncio.run(with_pipe(body))


class BlockingSaveService:
    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()

    async def __call__(self, form, use_now, set_default):
        self.calls += 1
        await self.release.wait()
        return SaveResult(["ok"])


def test_saving_ignores_repeat_submit_and_cancel(config):
    async def body(pipe):
        save = BlockingSaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        assert save.calls == 1 and app.saving
        await send(pipe, "\x13")
        await send(pipe, "\x03")
        await send(pipe, "\x1b")
        assert save.calls == 1
        assert app.app.is_running
        save.release.set()
        result = await finish(task)
        assert result is not None and save.calls == 1

    asyncio.run(with_pipe(body))


def test_api_format_cycles_and_updates_the_request_preview(config):
    async def body(pipe):
        app, _output = make_app(pipe, config, initial=draft(base_url="https://g.test"), save=SaveService())
        task = await start(app)
        await send(pipe, "\x1b[Z")
        assert app.app.layout.has_focus(app.api_selector.control)
        expected = [
            ("anthropic-messages", "https://g.test/v1/messages"),
            ("openai-responses", "https://g.test/responses"),
            ("openai-completions", "https://g.test/chat/completions"),
        ]
        for api, url in expected:
            await send(pipe, "\x1b[C")
            assert app.api_selector.options[app.api_selector.index][0] == api
            assert app._validation.request_url == url
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_full_endpoint_suggestion_requires_the_button(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(
            pipe, config, initial=draft(base_url="https://g.test/v1/chat/completions"), save=save)
        task = await start(app)
        assert app._validation.suggestion.suggested_root == "https://g.test/v1"
        await send(pipe, "\x13")
        assert save.calls == []
        assert "base_url" in app.draft.errors
        assert app.app.layout.has_focus(app.inputs["base_url"].control)
        app._apply_root()
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None
        assert result.form.base_url == "https://g.test/v1"

    asyncio.run(with_pipe(body))


def test_protocol_mismatch_offers_a_same_page_correction(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(
            pipe, config, initial=draft(base_url="https://g.test/v1/messages"), save=save)
        task = await start(app)
        assert app._validation.suggestion.matching_api == "anthropic-messages"
        await send(pipe, "\x13")
        assert save.calls == []
        assert "api" in app.draft.touched
        app._apply_format()
        assert app.api_selector.options[app.api_selector.index][0] == "anthropic-messages"
        app._apply_root()
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None
        assert result.form.api == "anthropic-messages"
        assert result.form.base_url == "https://g.test"

    asyncio.run(with_pipe(body))


def test_plain_http_needs_the_confirmation_checkbox(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(
            pipe, config, initial=draft(base_url="http://g.test/v1"), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        assert save.calls == []
        assert "allow_http" in app.draft.errors
        assert app.app.layout.has_focus(app.checkboxes["allow_http"].control)
        app.checkboxes["allow_http"].checked = True
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None and len(save.calls) == 1

    asyncio.run(with_pipe(body))


def test_local_http_needs_no_confirmation(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(
            pipe, config, initial=draft(base_url="http://localhost:8000/v1"), save=save)
        task = await start(app)
        await send(pipe, "\x13")
        result = await finish(task)
        assert result is not None and len(save.calls) == 1

    asyncio.run(with_pipe(body))


def test_narrow_terminal_keeps_the_form_and_the_draft(config):
    async def body(pipe):
        app, output = make_app(pipe, config, initial=draft(), width=70, height=24)
        task = await start(app)
        assert not app._too_small()
        assert not app._wide()
        await send(pipe, "\t")
        assert app.app.layout.has_focus(app.inputs["base_url"].control)
        output.size = Size(15, 50)
        app.app.invalidate()
        await asyncio.sleep(0.05)
        assert app._too_small()
        assert app.inputs["api_key"].text == "secret-key"
        output.size = Size(24, 80)
        app.app.invalidate()
        await asyncio.sleep(0.05)
        assert not app._too_small()
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_standard_size_renders_the_full_card(config):
    async def body(pipe):
        app, _output = make_app(pipe, config, initial=draft(), width=100, height=36)
        task = await start(app)
        screen = app.app.renderer.last_rendered_screen
        assert screen is not None and app._columns() == 100
        row = screen.data_buffer[0]
        line = "".join(row.get(x).char if row.get(x) else " " for x in range(100))
        assert line.startswith("  ╭─ Add custom API model ")
        assert line[2] == "╭" and line[97] == "╮"  # card is 96 wide, centered
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_small_terminal_scrolls_to_the_last_field(config):
    async def body(pipe):
        app, _output = make_app(pipe, config, initial=draft(), width=60, height=18)
        task = await start(app)
        assert app._scroll.vertical_scroll == 0
        for _ in range(10):
            await send(pipe, "\t")
        assert app._focus_key(app.app.layout.current_window) == "checkbox:set_default"
        assert app._scroll.vertical_scroll > 0
        await send(pipe, "\x1b")
        await finish(task)

    asyncio.run(with_pipe(body))


def test_tiny_terminal_shows_the_resize_message_and_escape_still_cancels(config):
    async def body(pipe):
        app, _output = make_app(pipe, config, initial=draft(), width=50, height=15)
        task = await start(app)
        assert app._too_small()
        texts = [w.content.text for w in app.app.layout.visible_windows
                 if hasattr(w.content, "text")]
        assert any("Resize terminal" in str(text) for text in texts)
        await send(pipe, "\x1b")
        assert await finish(task) is None

    asyncio.run(with_pipe(body))


def test_save_shortcut_is_configurable(config):
    async def body(pipe):
        save = SaveService()
        app, _output = make_app(pipe, config, initial=draft(), save=save, shortcuts=["c-x"])
        task = await start(app)
        await send(pipe, "\x13")
        assert save.calls == []
        await send(pipe, "\x18")
        result = await finish(task)
        assert result is not None and len(save.calls) == 1

    asyncio.run(with_pipe(body))


def test_draft_repr_hides_the_key():
    assert "secret-key" not in repr(FormDraft(api_key="secret-key"))
    assert "secret-key" not in repr(CustomApiForm(api_key="secret-key"))


# ---------------------------------------------------------------------------
# Whole-form validation
# ---------------------------------------------------------------------------


def test_validate_form_values_builds_the_form(config):
    result = validate_form_values({
        "api": "openai-completions", "provider_id": "g",
        "base_url": "https://g.test/v1", "api_key": "  k  ",
        "model_id": "vendor/model-a", "display_name": "",
        "context_window": "64000", "max_tokens": "8192",
        "image_input": True, "reasoning": True,
    }, {}, config)
    assert result.errors == {}
    assert result.form is not None
    assert result.form.display_name == "vendor/model-a"
    assert result.form.api_key == "k"
    assert result.form.context_window == 64000
    assert result.form.image_input and result.form.reasoning
    assert result.request_url == "https://g.test/v1/chat/completions"


def test_validate_form_values_reports_every_bad_field(config):
    result = validate_form_values({
        "api": "nonsense", "provider_id": "Bad ID", "base_url": "g.test",
        "api_key": "line\nbreak", "model_id": "", "display_name": "",
        "context_window": "0", "max_tokens": "8192",
    }, {}, config)
    assert result.form is None
    assert set(result.errors) == {
        "api", "provider_id", "base_url", "api_key", "model_id",
        "context_window",
    }


def test_validate_form_values_uses_the_provider_snapshot(config):
    registry = ModelRegistry(config)
    write_json(config.agent_dir / "models.json", {"providers": {"taken": {"models": []}}})
    registry.reload()
    result = validate_form_values({
        "api": "openai-completions", "provider_id": "taken",
        "base_url": "https://g.test/v1", "api_key": "k", "model_id": "m",
        "display_name": "", "context_window": "32768", "max_tokens": "4096",
    }, registry.providers, config)
    assert "already exists" in result.errors["provider_id"]


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------


class FakeFormUI:
    """Stands in for TerminalUI: runs the injected save and returns its result."""

    def __init__(self, form, use_now=True, set_default=False):
        self.form = form
        self.use_now = use_now
        self.set_default = set_default
        self.validate_seen = False

    async def custom_api_form(self, validate, save):
        self.validate_seen = True
        outcome = await save(self.form, self.use_now, self.set_default)
        return CustomApiFormResult(self.form, self.use_now, self.set_default, outcome.messages)


def test_custom_login_with_ui_uses_the_page_and_applies_choices(config, capsys):
    registry = ModelRegistry(config)
    agent = Agent(resolve_model("faux"))
    session = SessionManager(cwd=str(config.cwd))
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1",
                         api_key="secret-key", model_id="vendor/model-a")
    ui = FakeFormUI(form, use_now=True, set_default=True)
    asyncio.run(custom_login_command(registry, agent=agent, session=session, ui=ui))
    assert ui.validate_seen
    output = capsys.readouterr().out
    assert "saved provider 'my-gateway'" in output
    assert "secret-key" not in output
    assert "connection has not been verified" in output
    assert "default model: my-gateway/vendor/model-a" in output
    assert "model: my-gateway/vendor/model-a" in output
    assert agent.model.provider == "my-gateway"
    settings = read_json(config.agent_dir / "settings.json")
    assert settings["provider"] == "my-gateway" and settings["model"] == "vendor/model-a"
    for path in Path(config.cwd, ".pi", "sessions").glob("*.jsonl"):
        assert "secret-key" not in path.read_text(encoding="utf-8")


def test_save_never_contacts_the_provider(config, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("save must not send a request")

    monkeypatch.setattr(httpx.AsyncClient, "send", explode)
    registry = ModelRegistry(config)
    agent = Agent(resolve_model("faux"))
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1",
                         api_key="secret-key", model_id="vendor/model-a")
    ui = FakeFormUI(form, use_now=False, set_default=False)
    asyncio.run(custom_login_command(registry, agent=agent, ui=ui))
    assert agent.model.provider == "faux"


def test_custom_login_reports_both_unchanged_choices(config, capsys):
    registry = ModelRegistry(config)
    agent = Agent(resolve_model("faux"))
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1",
                         api_key="secret-key", model_id="vendor/model-a")
    asyncio.run(custom_login_command(registry, agent=agent, ui=FakeFormUI(form, False, False)))
    output = capsys.readouterr().out
    assert "default model: unchanged" in output
    assert "current model: unchanged" in output


def test_custom_login_with_ui_reports_partial_failure(config, capsys, monkeypatch):
    registry = ModelRegistry(config)
    agent = Agent(resolve_model("faux"))

    def boom(*args, **kwargs):
        raise OSError("settings locked")

    monkeypatch.setattr(registry.config, "save_many", boom)
    form = CustomApiForm(provider_id="my-gateway", base_url="https://g.test/v1",
                         api_key="secret-key", model_id="vendor/model-a")
    ui = FakeFormUI(form, use_now=False, set_default=True)
    asyncio.run(custom_login_command(registry, agent=agent, ui=ui))
    output = capsys.readouterr().out
    assert "setting the default failed" in output
    assert "connection has not been verified" in output
    assert "secret-key" not in output
