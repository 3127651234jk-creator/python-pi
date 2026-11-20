"""Faux provider usage estimate: prompt tokens and per-session cache split (F-AI-08)."""

import asyncio
import math
import os

from pi.ai.models import Model
from pi.ai.providers import faux, get_stream_fn
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import DoneEvent, TranscriptContext, UserMessage


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


def run(prompt, session_id=None, retention=None):
    async def drive():
        options = SimpleStreamOptions(session_id=session_id, cache_retention=retention)
        usage = None
        async for event in get_stream_fn("faux")(make_model(), TranscriptContext(messages=prompt), options):
            if isinstance(event, DoneEvent):
                usage = event.message.usage
        return usage

    return asyncio.run(drive())


def setup_function(_):
    # Enough turns for every request in one test; the script index advances
    # per request while the prompt cache persists across them.
    os.environ["FAUX_SCRIPT"] = "[" + ",".join(['{"text": "ok"}'] * 50) + "]"
    faux.reset_script_index()


def teardown_function(_):
    os.environ.pop("FAUX_SCRIPT", None)
    faux.reset_script_index()


def test_first_request_writes_the_whole_prompt_to_cache():
    usage = run([UserMessage(content="x" * 40, timestamp=0)], session_id="s1")
    assert usage.cache_read == 0
    assert usage.cache_write == usage.input


def test_second_request_same_session_reads_cache():
    prompt = [UserMessage(content="x" * 400, timestamp=0)]
    run(prompt, session_id="s2")
    grown = prompt + [UserMessage(content="y" * 40, timestamp=1)]
    usage = run(grown, session_id="s2")
    assert usage.cache_read > 0
    # The shared prefix is billed as cacheRead; input is the prompt estimate
    # minus what the cache covered; the remainder is cacheWrite.
    joined = "\n\n".join(f"user:{m.content}" for m in grown)
    prompt_tokens = math.ceil(len(joined) / 4)
    cached_chars = len(f"user:{'x' * 400}")  # the first message matched whole
    assert usage.cache_read == math.ceil(cached_chars / 4)
    assert usage.input == prompt_tokens - usage.cache_read
    assert usage.cache_write == math.ceil((len(joined) - cached_chars) / 4)


def test_prefix_compare_starts_at_the_first_differing_message():
    """A differing first message shares only the serialized role prefix."""
    first = [UserMessage(content="x" * 400, timestamp=0)]
    run(first, session_id="s6")
    other = [UserMessage(content="z" * 400, timestamp=0)]
    usage = run(other, session_id="s6")
    assert usage.cache_read == math.ceil(len("user:") / 4)  # the "user:" prefix
    assert usage.cache_write == usage.input


def test_other_session_starts_fresh():
    run([UserMessage(content="x" * 400, timestamp=0)], session_id="s3")
    usage = run([UserMessage(content="x" * 400, timestamp=0)], session_id="s4")
    assert usage.cache_read == 0 and usage.cache_write == usage.input


def test_none_retention_skips_caching():
    run([UserMessage(content="x" * 400, timestamp=0)], session_id="s5", retention="none")
    usage = run([UserMessage(content="x" * 400, timestamp=0)], session_id="s5", retention="none")
    assert usage.cache_read == 0 and usage.cache_write == 0
