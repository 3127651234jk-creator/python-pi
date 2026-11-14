import json

from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.diagnostics import Diagnostic, deduplicate_diagnostics, settings_diagnostics


def test_deduplicate_keeps_first_occurrence():
    diagnostics = deduplicate_diagnostics([
        Diagnostic("warning", "same"),
        Diagnostic("warning", "same"),
        Diagnostic("error", "same"),
        Diagnostic("error", "other"),
    ])
    assert [(item.type, item.message) for item in diagnostics] == [
        ("warning", "same"), ("error", "same"), ("error", "other")]


def test_settings_diagnostics_describes_paths():
    diagnostics = settings_diagnostics([("c:/p/.pi/settings.json", "bad"), ("", "conflict")])
    assert diagnostics[0].message == "Invalid settings file c:/p/.pi/settings.json: bad"
    assert diagnostics[1].message == "Invalid merged settings: conflict"


def test_invalid_settings_file_does_not_abort_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    (tmp_path / "user" / "settings.json").write_text("{broken", encoding="utf-8")
    config = Configuration(str(tmp_path))
    assert config.errors and "Cannot read configuration" in config.errors[0][1]
    assert config.settings == {}


def test_invalid_project_file_keeps_user_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    write_json(tmp_path / "user" / "settings.json", {"provider": "anthropic"})
    (tmp_path / ".pi").mkdir()
    (tmp_path / ".pi" / "settings.json").write_text(json.dumps({"theme": ""}), encoding="utf-8")
    config = Configuration(str(tmp_path))
    assert config.errors, "project validation failure must be collected"
    assert config.settings.get("provider") == "anthropic"
    assert "theme" not in config.settings


def test_new_settings_keys_are_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    config = Configuration(str(tmp_path))
    config.save("default_project_trust", "never")
    config.save("enable_install_telemetry", False)
    config.save("cache_retention", "1h")
    assert config.settings["default_project_trust"] == "never"
    assert config.settings["enable_install_telemetry"] is False
    assert config.settings["cache_retention"] == "1h"


def test_new_settings_keys_are_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    config = Configuration(str(tmp_path))
    for key, value in (("default_project_trust", "sometimes"),
                       ("enable_install_telemetry", "yes"),
                       ("cache_retention", "30m")):
        try:
            config.save(key, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {key}")
