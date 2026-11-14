import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pi.coding_agent import cli


def test_browser_login_cancels_prompt_without_executor_or_enter(tmp_path, monkeypatch):
    saved = Mock()
    registry = SimpleNamespace(providers={"openai": {}},
                               config=SimpleNamespace(agent_dir=tmp_path),
                               auth=SimpleNamespace(set=saved))
    with create_pipe_input() as terminal:
        session = PromptSession(input=terminal, output=DummyOutput())
        monkeypatch.setattr(cli, "PromptSession", lambda: session)

        async def browser_login(notify, prompt, device_id):
            manual = asyncio.create_task(prompt("Paste callback URL: "))
            try:
                while not session.app.is_running:
                    await asyncio.sleep(0)
                # The browser callback wins without sending any terminal input.
                return {"type": "oauth", "access": "test-token"}
            finally:
                manual.cancel()
                await asyncio.gather(manual, return_exceptions=True)

        monkeypatch.setattr(cli, "login_openai_chatgpt", browser_login)

        async def scenario():
            monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("blocking input")))
            monkeypatch.setattr(cli.getpass, "getpass", Mock(side_effect=AssertionError("blocking getpass")))
            await asyncio.wait_for(cli.authenticate(registry, "openai", oauth=True), 2)
            assert not session.app.is_running
            # Cancellation must release stdin so the next prompt can read it.
            next_prompt = asyncio.create_task(session.prompt_async("Next: "))
            terminal.send_text("next\n")
            assert await asyncio.wait_for(next_prompt, 2) == "next"

        asyncio.run(scenario())
    saved.assert_called_once_with("openai", {"type": "oauth", "access": "test-token"})
