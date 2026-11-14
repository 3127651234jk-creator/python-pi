import asyncio
import shutil
import sys

import pytest

from pi.coding_agent.tools import bash
from pi.coding_agent.tools.powershell import create_powershell_tool


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell stderr serialization")
@pytest.mark.parametrize("executable", ["powershell", "pwsh"])
@pytest.mark.parametrize("command,expected,exit_code", [
    ("Write-Error 'readable failure'; exit 7", "readable failure", 7),
    ("Get-Item -LiteralPath './missing-pi-regression-file'; exit 3", "missing-pi-regression-file", 3),
    ("$value = '中文';\nWrite-Output ($value + ' \"quoted\"'); exit 0", '中文 "quoted"', 0),
])
def test_powershell_redirected_errors_are_plain_text(tmp_path, monkeypatch, executable, command, expected, exit_code):
    path = shutil.which(executable)
    if path is None:
        pytest.skip(f"{executable} unavailable")
    monkeypatch.setattr(bash.shutil, "which", lambda name: path)
    updates = []

    async def scenario():
        async def update(result):
            updates.append(result.content[0].text)

        return await create_powershell_tool(str(tmp_path)).execute(
            "test", {"command": command, "timeout": 10}, on_update=update)

    result = asyncio.run(scenario())
    output = result.structured_content["output"]
    assert expected in output
    assert "#< CLIXML" not in output
    assert "<Objs" not in output
    assert all("#< CLIXML" not in text for text in updates)
    assert result.structured_content["exit_code"] == exit_code
    assert result.is_error == (exit_code != 0)
