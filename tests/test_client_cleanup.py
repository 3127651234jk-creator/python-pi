"""Client cleanup regressions."""

from __future__ import annotations

import asyncio

import pytest

from pi.client import Client


def test_close_clears_state_when_transport_close_fails():
    class BrokenWriter:
        def close(self):
            return None

        async def wait_closed(self):
            raise ConnectionResetError("connection reset")

    async def main():
        client = Client("server")
        client.writer = BrokenWriter()
        client.attachment = {"serverId": "server"}
        client.reader_task = asyncio.create_task(asyncio.Event().wait())

        with pytest.raises(ConnectionResetError, match="connection reset"):
            await client.close()

        assert client.writer is None
        assert client.reader_task is None
        assert client.attachment is None

    asyncio.run(main())
