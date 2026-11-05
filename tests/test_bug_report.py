import asyncio
import json
import zipfile

import httpx
import pytest

from pi.ai.models import resolve_model
from pi.ai.providers.faux import reset_script_index
from pi.ai.types import AssistantMessage, Usage, UserMessage
from pi.coding_agent.bug_report import (bug_report_archive_file_name, bug_report_files,
                                        collect_bug_report_diagnostics, collect_bug_report_metadata,
                                        generate_bug_report_summary, is_sensitive_key,
                                        redact_json_value, redact_url, upload_bug_report,
                                        write_bug_report_archive)
from pi.coding_agent.session import SessionManager


@pytest.fixture(autouse=True)
def _reset_faux_index():
    yield
    reset_script_index()


def run(coro):
    return asyncio.run(coro)


def test_is_sensitive_key():
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("apiKey")
    assert is_sensitive_key("AUTHORIZATION")
    assert is_sensitive_key("x-api-key")
    assert not is_sensitive_key("model")
    assert not is_sensitive_key("tokens")


def test_redact_url_credentials_and_query():
    redacted = redact_url("https://user:pass@example.com/v1?api_key=abc&limit=5")
    assert redacted.startswith("https://example.com/v1?")
    assert "api_key=%3Credacted%3E" in redacted or "api_key=<redacted>" in redacted
    assert "limit=5" in redacted
    assert "user:pass" not in redacted


def test_redact_url_keeps_plain_urls():
    assert redact_url("https://api.example.com/v1/messages") == "https://api.example.com/v1/messages"


def test_redact_json_value_nested():
    value = redact_json_value({
        "api_key": "sk-1",
        "nested": {"refreshToken": "t", "url": "https://h.com/?token=1"},
        "list": ["https://a.dev/?ok=1"],
        "number": 3,
    })
    assert value["api_key"] == "<redacted>"
    assert value["nested"]["refreshToken"] == "<redacted>"
    assert "token=" in value["nested"]["url"] and "1" not in value["nested"]["url"].split("token=")[1]
    assert value["list"] == ["https://a.dev/?ok=1"]
    assert value["number"] == 3


def test_collect_metadata_redacts_settings():
    metadata = collect_bug_report_metadata(
        hint="crash on start", session_id="s1", cwd=".", include_session=False,
        include_summary=True, message_count=3, model=None, thinking_level="off",
        extensions=["/x/extension.py"], extension_errors=[],
        global_settings={"api_key": "sk", "provider": "anthropic"},
        project_settings={}, id="fixed-id")
    assert metadata["id"] == "fixed-id"
    assert metadata["hint"] == "crash on start"
    assert metadata["session"]["included"] is False
    assert metadata["settings"]["global"]["api_key"] == "<redacted>"
    assert metadata["settings"]["global"]["provider"] == "anthropic"
    assert "piEnvironmentVariables" in metadata["environment"]


def test_collect_diagnostics_only_failed_turns(tmp_path):
    session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    ok = AssistantMessage(content=[], api="faux", provider="faux", model="faux",
                          usage=Usage(), stop_reason="stop", timestamp=1)
    bad = AssistantMessage(content=[], api="faux", provider="faux", model="faux",
                           usage=Usage(), stop_reason="error", error_message="boom", timestamp=2)
    session.append(UserMessage(content=[{"type": "text", "text": "hi"}], timestamp=0))
    session.append(ok)
    session.append(bad)
    report = collect_bug_report_diagnostics(session, crashes=[{"message": "old", "notified": True}])
    assert report["assistantMessageCount"] == 2
    assert len(report["assistant"]) == 1
    assert report["assistant"][0]["errorMessage"] == "boom"
    assert report["crashes"] == [{"message": "old"}]


def test_archive_contains_expected_files(tmp_path):
    bundle = {"metadata": {"id": "id1"}, "diagnostics": {"assistant": []},
              "sessionJsonl": '{"type":"message"}\n', "summary": "# report"}
    path = tmp_path / bug_report_archive_file_name("id1")
    assert write_bug_report_archive(bundle, str(path)) == str(path)
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["diagnostics.json", "report.json", "session.jsonl", "summary.md"]
        assert json.loads(archive.read("report.json"))["id"] == "id1"


def test_files_skip_missing_parts():
    bundle = {"metadata": {}, "diagnostics": {}}
    assert [name for name, _type, _data in bug_report_files(bundle)] == ["report.json", "diagnostics.json"]


def test_generate_summary_with_faux(monkeypatch):
    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([{"text": "Report body."}]))
    reset_script_index()
    model = resolve_model("faux")
    messages = [UserMessage(content=[{"type": "text", "text": "hello"}], timestamp=0)]
    assert run(generate_bug_report_summary(messages, model, hint="it broke")) == "Report body."


def test_generate_summary_rejects_empty(monkeypatch):
    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([{"text": "  "}]))
    reset_script_index()
    model = resolve_model("faux")
    try:
        run(generate_bug_report_summary([], model))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_upload_success(monkeypatch):
    async def post(self, url, files=None, headers=None, **kwargs):
        assert url.endswith("/v1/bug-reports")
        assert headers is None
        return httpx.Response(200, json={"ok": True, "bug_report": {"id": "report-1"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    bundle = {"metadata": {"id": "x"}, "diagnostics": {}}
    assert run(upload_bug_report(bundle)) == "report-1"


def test_upload_with_token_and_failure(monkeypatch):
    seen = {}

    async def post(self, url, files=None, headers=None, **kwargs):
        seen["headers"] = headers
        return httpx.Response(500, json={"ok": False, "error": "server down"})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    bundle = {"metadata": {}, "diagnostics": {}}
    try:
        run(upload_bug_report(bundle, token="tok"))
    except ValueError as exc:
        assert "server down" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert seen["headers"] == {"Authorization": "Bearer tok"}
