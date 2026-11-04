"""Shared pytest helpers: asyncio runner and faux-agent fixtures."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(coro):
    return asyncio.run(coro)
