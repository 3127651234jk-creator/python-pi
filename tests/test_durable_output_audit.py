"""Streaming output must preserve UTF-8 across arbitrary byte chunks."""

import pytest

from pi.durable.output import OutputBuffer, OutputLimits


@pytest.mark.parametrize("retain", ["head", "tail"])
@pytest.mark.parametrize("text", ["café", "中文", "a🙂b", "\ufeffhello"])
def test_utf8_output_is_independent_of_chunk_boundaries(retain, text):
    encoded = text.encode("utf-8")
    for split in range(1, len(encoded)):
        output = OutputBuffer(OutputLimits(max_bytes=1000, max_lines=100, retain=retain))
        output.push(encoded[:split])
        output.snapshot()
        output.push(encoded[split:])
        output.end()
        result = output.snapshot()
        assert result.text == text
        assert result.dropped_bytes == result.dropped_lines == 0


@pytest.mark.parametrize("text", ["é", "中", "🙂"])
def test_complete_utf8_character_is_visible_before_stream_end(text):
    output = OutputBuffer(OutputLimits(max_bytes=1000, max_lines=100, retain="tail"))
    output.push(text.encode("utf-8"))
    assert output.snapshot().text == text
