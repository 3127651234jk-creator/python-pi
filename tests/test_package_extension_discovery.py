import asyncio
import json

import pytest

from pi.agent.agent import Agent
from pi.ai.models import Model
from pi.coding_agent.config import Configuration
from pi.coding_agent.extensions import ExtensionManager
from pi.coding_agent.packages import PackageManager


@pytest.mark.parametrize("entries", [["index.js"], ["first.py", "second.py"],
                                      ["extensions"], ["bundle"]])
def test_installed_manifest_extensions_are_discovered(tmp_path, monkeypatch, entries):
    source = tmp_path / "source"
    source.mkdir()
    (source / "package.json").write_text(json.dumps({"pi": {"extensions": entries}}))
    expected_names = []
    for entry in entries:
        path = source / entry
        if path.suffix:
            path.write_text("// extension" if path.suffix == ".js" else "def setup(api): pass\n")
            expected_names.append(path.name)
        else:
            path.mkdir()
            name = "index.js" if entry == "bundle" else "example.py"
            (path / name).write_text("// extension" if name.endswith("js") else "def setup(api): pass\n")
            expected_names.append(name)
    config = Configuration(tmp_path, tmp_path / "agent")
    packages = PackageManager(config)

    async def install_npm(parsed, notify):
        return "plugin", "1.0.0", source

    monkeypatch.setattr(packages, "_install_npm", install_npm)
    asyncio.run(packages.install("npm:plugin", lambda text: None))
    extensions = ExtensionManager(config, Agent(Model("faux", "Faux", "faux", "faux", "")))
    paths = extensions.paths()
    assert sorted(path.name for path in paths) == sorted(expected_names)
    assert len(paths) == len(set(paths))
    assert all(path.is_relative_to(config.agent_dir / "extensions") for path in paths)
    config.settings["include_default_extensions"] = False
    assert extensions.paths() == []
