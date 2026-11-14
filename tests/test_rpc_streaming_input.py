"""Streaming prompts must queue even when an extension manager is present."""

import asyncio
import io
import json

import pytest

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.config import Configuration
from pi.coding_agent.extensions import ExtensionManager
from pi.coding_agent.rpc import JsonWriter, RpcServer


@pytest.mark.parametrize("behavior,queue", [("steer", "_steering_queue"), ("followUp", "_follow_up_queue")])
def test_busy_prompt_queues_with_extension_manager(tmp_path, monkeypatch, behavior, queue):
    async def scenario():
        agent = Agent(resolve_model("faux"))
        extensions = ExtensionManager(Configuration(str(tmp_path), str(tmp_path / "agent")), agent)

        async def process_input(text):
            pytest.fail("busy input must not run extension commands")

        monkeypatch.setattr(extensions, "process_input", process_input)
        output = io.StringIO()
        server = RpcServer(agent, JsonWriter(output), extensions=extensions)
        server.task = asyncio.create_task(asyncio.Event().wait())
        try:
            await server.dispatch({"id": "queued", "type": "prompt", "message": "next",
                                   "streamingBehavior": behavior,
                                   "images": [{"type": "image", "data": "abc", "mimeType": "image/png"}]})
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            assert records[-1] == {"id": "queued", "type": "response", "command": "prompt",
                                   "success": True, "data": {"disposition": "queued"}}
            assert records[0]["type"] == "queue_update"
            assert records[0]["steering" if behavior == "steer" else "followUp"] == ["next"]
            messages = getattr(agent, queue)
            assert len(messages) == 1
            assert messages[0].content[1].data == "abc"
            await server.dispatch({"type": "prompt", "message": "rejected"})
            response = json.loads(output.getvalue().splitlines()[-1])
            assert response["success"] is False
            assert "streamingBehavior" in response["error"]
            assert len(messages) == 1
        finally:
            server.task.cancel()
            await asyncio.gather(server.task, return_exceptions=True)
            await extensions.close()

    asyncio.run(scenario())
