"""User/project settings and atomic JSON persistence."""

import json
import os
import tempfile
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read configuration: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".pi-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_settings(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        result[key] = merge_settings(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


class Configuration:
    def __init__(self, cwd: str, agent_dir: str = "") -> None:
        self.cwd = Path(cwd)
        self.agent_dir = Path(agent_dir or os.environ.get("PI_PY_AGENT_DIR") or Path.home() / ".pi-python")
        self.project_trusted = True  # set by the CLI after trust resolution
        self.errors = []  # [(path, message)] for settings files that failed to load
        self.settings: dict = {}
        self.reload()

    def reload(self) -> None:
        """Merge user and project settings; collect per-file errors and keep
        the last valid merged result so a bad file cannot abort startup."""
        self.errors = []
        merged: dict = {}
        for path, project in ((self.agent_dir / "settings.json", False), (self.cwd / ".pi" / "settings.json", True)):
            if project and not self.project_trusted:
                continue
            try:
                data = read_json(path)
            except ValueError as exc:
                self.errors.append((str(path), str(exc)))
                continue
            try:
                validate_settings(data)
            except ValueError as exc:
                self.errors.append((str(path), str(exc)))
                continue
            merged = merge_settings(merged, data)
        try:
            validate_settings(merged)
        except ValueError as exc:
            # Only cross-file conflicts land here; report them like file errors.
            self.errors.append(("", str(exc)))
            return
        self.settings = merged

    def save(self, key: str, value: object, project: bool = False) -> None:
        self.save_many({key: value}, project)

    def save_many(self, values: dict, project: bool = False) -> None:
        """Persist several settings keys as one file update."""
        path = (self.cwd / ".pi" if project else self.agent_dir) / "settings.json"
        settings = read_json(path)
        settings.update(values)
        validate_settings(settings)
        write_json(path, settings)
        self.reload()


def validate_settings(settings: dict) -> None:
    allowed = {"provider", "model", "thinking", "session_dir", "compaction", "skills", "prompts", "include_default_resources", "enable_skill_commands"}
    allowed.update({"extensions", "include_default_extensions"})
    allowed.update({"theme", "keybindings"})
    allowed.update({"packages", "sync_catalog", "markdown", "fold_output", "codemode"})
    allowed.update({"images"})
    allowed.update({"default_project_trust", "enable_install_telemetry", "cache_retention"})
    if set(settings) - allowed:
        raise ValueError("Unknown settings key; supported: " + ", ".join(sorted(allowed)))
    if settings.get("theme", "dark") not in ("dark", "light", "mono"):
        raise ValueError("theme must be dark, light or mono")
    bindings = settings.get("keybindings", {})
    if not isinstance(bindings, dict) or set(bindings) - {"submit", "newline", "abort", "models", "sessions", "exit", "complete", "save"}:
        raise ValueError("Invalid keybindings actions")
    if not all(isinstance(keys, list) and keys and all(isinstance(key, str) and key.strip() for key in keys) for keys in bindings.values()):
        raise ValueError("Each keybindings action requires a nonempty list of shortcuts")
    for key in ("skills", "prompts", "extensions"):
        if key in settings and (not isinstance(settings[key], list) or not all(isinstance(path, str) and path for path in settings[key])):
            raise ValueError(f"{key} must be a list of resource paths")
    if "packages" in settings:
        packages = settings["packages"]
        if not isinstance(packages, list) or not all(
                isinstance(entry, str) and entry or (isinstance(entry, dict) and isinstance(entry.get("source"), str))
                for entry in packages):
            raise ValueError("packages must be a list of source strings or {source} objects")
    for key in ("sync_catalog",):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "markdown" in settings and type(settings["markdown"]) is not bool:
        raise ValueError("markdown must be boolean")
    images = settings.get("images", {})
    if not isinstance(images, dict) or set(images) - {"auto_resize", "block_images"} or not all(
            type(value) is bool for value in images.values()):
        raise ValueError("images settings must be {auto_resize?: bool, block_images?: bool}")
    if "fold_output" in settings and (type(settings["fold_output"]) is not int or settings["fold_output"] < 0):
        raise ValueError("fold_output must be a nonnegative integer")
    if "codemode" in settings:
        codemode = settings["codemode"]
        if not isinstance(codemode, dict) or set(codemode) - {"enabled", "timeout_ms"} or (
                "enabled" in codemode and type(codemode["enabled"]) is not bool) or (
                "timeout_ms" in codemode and (type(codemode["timeout_ms"]) is not int or codemode["timeout_ms"] <= 0)):
            raise ValueError("codemode settings must be {enabled?: bool, timeout_ms?: int}")
    for key in ("include_default_resources", "enable_skill_commands", "include_default_extensions"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("provider", "model", "session_dir"):
        if key in settings and (not isinstance(settings[key], str) or not settings[key]):
            raise ValueError(f"{key} must be a nonempty string")
    if "default_project_trust" in settings and settings["default_project_trust"] not in ("ask", "always", "never"):
        raise ValueError("default_project_trust must be ask, always or never")
    if "enable_install_telemetry" in settings and type(settings["enable_install_telemetry"]) is not bool:
        raise ValueError("enable_install_telemetry must be boolean")
    if "cache_retention" in settings and settings["cache_retention"] not in ("5m", "1h", False):
        raise ValueError("cache_retention must be 5m, 1h or false")
    if settings.get("thinking", "off") not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Invalid thinking level")
    compaction = settings.get("compaction", {})
    if not isinstance(compaction, dict) or set(compaction) - {"enabled", "reserve_tokens", "keep_recent_tokens", "summary_tokens"}:
        raise ValueError("Invalid compaction settings")
    for key, value in compaction.items():
        if key == "enabled":
            if type(value) is not bool:
                raise ValueError(f"Invalid compaction setting: {key}")
        elif key == "summary_tokens":
            if type(value) is not int or value < 0:  # 0 means "derive from reserve_tokens"
                raise ValueError(f"Invalid compaction setting: {key}")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"Invalid compaction setting: {key}")
