import json

import pytest

from pi.coding_agent.config import Configuration, validate_settings
from pi.coding_agent.tui import load_themes


def test_user_theme_saves_and_reloads_without_dropping_settings(tmp_path):
    agent_dir = tmp_path / "agent"
    themes = agent_dir / "themes"
    themes.mkdir(parents=True)
    (themes / "custom.json").write_text(json.dumps({"prompt": "bold ansired"}))
    config = Configuration(tmp_path, agent_dir)
    assert "custom" in load_themes(agent_dir)
    config.save_many({"theme": "custom", "model": "preserved"})
    reloaded = Configuration(tmp_path, agent_dir)
    assert reloaded.errors == []
    assert reloaded.settings["theme"] == "custom"
    assert reloaded.settings["model"] == "preserved"


def test_manually_configured_theme_does_not_reject_file(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"theme": "my-theme", "markdown": False}))
    config = Configuration(tmp_path, agent_dir)
    assert config.errors == []
    assert config.settings == {"theme": "my-theme", "markdown": False}


@pytest.mark.parametrize("value", [None, False, 1, [], {}, "", "  "])
def test_invalid_theme_setting_rejected(value):
    with pytest.raises(ValueError, match="theme must be a nonempty string"):
        validate_settings({"theme": value})
