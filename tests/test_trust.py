import asyncio

from pi.coding_agent.trust import (TrustStore, has_trust_requiring_project_resources,
                                   project_trust_parent_path, resolve_project_trusted, trust_options)


def run(coro):
    return asyncio.run(coro)


def test_store_roundtrip_and_nearest_ancestor(tmp_path):
    store = TrustStore(tmp_path)
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    assert store.get(str(child)) is None
    store.set(str(parent), True)
    assert store.get(str(child)) is True
    store.set(str(child), False)
    assert store.get(str(child)) is False
    store.set(str(child), None)
    assert store.get(str(child)) is True  # falls back to the parent entry


def test_store_file_is_sorted_json(tmp_path):
    store = TrustStore(tmp_path)
    store.set(str(tmp_path / "b"), False)
    store.set(str(tmp_path / "a"), True)
    lines = (tmp_path / "trust.json").read_text(encoding="utf-8").strip().splitlines()
    keys = [line.strip().split(":")[0].strip('"') for line in lines[1:-1]]
    assert keys == sorted(keys)


def test_has_trust_requiring_resources(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    assert not has_trust_requiring_project_resources(str(project), home=tmp_path / "home")
    (project / ".pi" / "extensions").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(project), home=tmp_path / "home")


def test_ancestor_agents_skills_requires_trust(tmp_path):
    root = tmp_path / "repo"
    (root / ".agents" / "skills").mkdir(parents=True)
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(nested), home=tmp_path / "home")


def test_user_agents_skills_is_excluded(tmp_path):
    home = tmp_path / "home"
    (home / ".agents" / "skills").mkdir(parents=True)
    assert not has_trust_requiring_project_resources(str(home), home=home)


def test_parent_path(tmp_path):
    assert project_trust_parent_path(str(tmp_path / "proj")) == str(tmp_path)
    assert project_trust_parent_path(str(tmp_path.anchor)) is None


def test_options_cover_session_only_variants(tmp_path):
    labels = [option["label"] for option in trust_options(str(tmp_path))]
    assert labels[0] == "Trust"
    assert f"Trust parent folder ({tmp_path.parent})" in labels
    assert "Trust (this session only)" in labels
    assert "Do not trust" in labels
    assert "Do not trust (this session only)" in labels


def test_override_wins(tmp_path):
    store = TrustStore(tmp_path)
    assert run(resolve_project_trusted(str(tmp_path), store, None, override=True)) is True
    assert run(resolve_project_trusted(str(tmp_path), store, None, override=False)) is False


def test_no_resources_is_trusted_without_asking(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(resolve_project_trusted(str(empty), TrustStore(tmp_path), None)) is True


def test_stored_decision_skips_prompt(tmp_path):
    store = TrustStore(tmp_path)
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)
    store.set(str(tmp_path), False)
    assert run(resolve_project_trusted(str(tmp_path), store, None)) is False


def test_default_setting(tmp_path):
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)
    store = TrustStore(tmp_path)
    assert run(resolve_project_trusted(str(tmp_path), store, None, settings={"default_project_trust": "always"})) is True
    assert run(resolve_project_trusted(str(tmp_path), store, None, settings={"default_project_trust": "never"})) is False


def test_no_ui_falls_back_to_untrusted(tmp_path):
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)
    notified = []
    assert run(resolve_project_trusted(str(tmp_path), TrustStore(tmp_path), None, notify=notified.append)) is False
    assert notified and "trust" in notified[0].lower()


def test_dialog_trust_session_only_does_not_persist(tmp_path):
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)
    store = TrustStore(tmp_path)
    options = trust_options(str(tmp_path))
    chosen = next(option for option in options if option["label"] == "Trust (this session only)")

    async def choose(title, values):
        return dict(values)[chosen["label"]]

    assert run(resolve_project_trusted(str(tmp_path), store, choose, has_ui=True)) is True
    assert store.get(str(tmp_path)) is None


def test_dialog_do_not_trust_persists(tmp_path):
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)
    store = TrustStore(tmp_path)
    options = trust_options(str(tmp_path))
    chosen = next(option for option in options if option["label"] == "Do not trust")

    async def choose(title, values):
        return dict(values)[chosen["label"]]

    assert run(resolve_project_trusted(str(tmp_path), store, choose, has_ui=True)) is False
    assert store.get(str(tmp_path)) is False


def test_cancelled_dialog_is_untrusted(tmp_path):
    (tmp_path / ".pi" / "extensions").mkdir(parents=True)

    async def choose(title, values):
        return None

    assert run(resolve_project_trusted(str(tmp_path), TrustStore(tmp_path), choose, has_ui=True)) is False
