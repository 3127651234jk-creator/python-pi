"""Persistent crash log for uncaught exceptions (port of crash-log.ts).

Records live in ``<agent_dir>/crashes.json``; the newest 5 are kept and
unnotified records from the last 7 days are reported once on the next
interactive startup. ``record_crash`` is best effort: it never raises from a
call site that is already crashing.
"""

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

MAX_CRASH_RECORDS = 5
MAX_AGE_SECONDS = 7 * 24 * 60 * 60
KINDS = ("uncaught_exception", "fatal_error")


def crash_log_path(agent_dir=None) -> Path:
    directory = Path(agent_dir) if agent_dir else Path(os.environ.get("PI_PY_AGENT_DIR") or Path.home() / ".pi-python")
    return directory / "crashes.json"


def read_crash_log(path=None) -> List[dict]:
    path = Path(path) if path else crash_log_path()
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(records, list):
        return []
    return [record for record in records
            if isinstance(record, dict) and isinstance(record.get("timestamp"), str)
            and isinstance(record.get("message"), str)]


def _write(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_crash(kind: str, error: BaseException, cwd: str, session_file: Optional[str] = None,
                 agent_dir=None, path=None) -> Optional[dict]:
    """Append one crash record; returns it, or None when persistence failed."""
    if kind not in KINDS:
        raise ValueError(f"Crash kind must be one of {KINDS}")
    try:
        path = Path(path) if path else crash_log_path(agent_dir)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": pi_version(),
            "kind": kind,
            "message": str(error) or error.__class__.__name__,
            "stack": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "sessionFile": session_file,
            "cwd": cwd,
        }
        _write((read_crash_log(path) + [record])[-MAX_CRASH_RECORDS:], path)
        return record
    except Exception:  # noqa: BLE001 - best effort while crashing
        return None


def take_unnotified_crash(path=None) -> Optional[dict]:
    """Newest recent unnotified crash, marking pending records as announced."""
    path = Path(path) if path else crash_log_path()
    records = read_crash_log(path)
    now = datetime.now(timezone.utc).timestamp()
    crash = None
    for record in reversed(records):
        try:
            age = now - datetime.fromisoformat(record["timestamp"]).timestamp()
        except ValueError:
            continue
        if not record.get("notified") and age <= MAX_AGE_SECONDS:
            crash = record
            break
    if crash is None:
        return None
    try:
        _write([{**record, "notified": True} if not record.get("notified") else record for record in records], path)
    except OSError:
        pass  # Showing the notice again is harmless.
    return crash


def clear_crash_log(path=None) -> None:
    path = Path(path) if path else crash_log_path()
    try:
        path.unlink()
    except OSError:
        pass


def pi_version() -> str:
    try:
        from importlib.metadata import version
        return version("pi-python")
    except Exception:  # noqa: BLE001 - not installed as a package
        return "unknown"


def find_extension_stack_matches(stack: Optional[str], extension_paths: List[str]) -> List[str]:
    """Extension paths whose source files appear in a Python traceback."""
    if not stack:
        return []
    normalized = stack.replace("\\", "/")
    matches = []
    for path in extension_paths:
        candidate = str(path).replace("\\", "/").rstrip("/")
        if candidate and candidate in normalized and candidate not in matches:
            matches.append(path)
    return matches
