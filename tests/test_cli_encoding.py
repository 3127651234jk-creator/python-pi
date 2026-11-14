"""Redirected text output must use UTF-8 even with a legacy Windows encoding."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_redirected_text_cli_uses_utf8(tmp_path):
    text = "Hello \U0001f600 \u4e2d\u6587"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PI_OFFLINE": "1",
           "PI_PY_AGENT_DIR": str(tmp_path / "agent"),
           "FAUX_SCRIPT": json.dumps([{"text": text}]),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    output = tmp_path / "out.txt"
    with output.open("wb") as stdout:
        result = subprocess.run([
            sys.executable, "-m", "pi", "--provider", "faux", "--no-session", "hello",
        ], cwd=tmp_path, env=env, stdout=stdout, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert text in output.read_bytes().decode("utf-8")
