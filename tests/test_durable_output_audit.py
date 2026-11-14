"""Streaming output must preserve UTF-8 across arbitrary byte chunks."""

import pytest

from pi.durable.output import OutputBuffer, OutputLimits, bound_output


@pytest.mark.parametrize("retain", ["head", "tail"])
@pytest.mark.parametrize("text", ["café", "中文", "a🙂b"])
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


def test_bound_output_keeps_a_feff_at_the_start_of_a_kept_slice():
    # Slices decode exactly: a U+FEFF at a slice's start is text, not a byte-order mark.
    sliced = bound_output("x\ufeffa", OutputLimits(max_bytes=4, max_lines=10, retain="tail"))
    assert sliced.text == "\ufeffa"
    assert sliced.dropped_bytes == 1
    assert sliced.dropped_lines == 0


def test_byte_order_mark_is_dropped_only_at_the_start_of_the_output():
    bom = b"\xef\xbb\xbf"
    output = OutputBuffer(OutputLimits(max_bytes=10, max_lines=100, retain="tail"))
    output.push(bom[:1])
    output.push(b"\xbb\xbf" + "a".encode("utf-8"))
    output.push("b")
    # A U+FEFF after a string chunk is text.
    output.push(bom + "c".encode("utf-8"))
    output.end()
    assert output.snapshot().text == "ab\ufeffc"


@pytest.mark.parametrize("retain", ["head", "tail"])
def test_byte_order_mark_at_the_start_of_byte_output_is_dropped(retain):
    # The mark is dropped whatever chunk boundary it crosses, since nothing is accepted before it.
    encoded = "\ufeffhello".encode("utf-8")
    for split in range(1, len(encoded)):
        output = OutputBuffer(OutputLimits(max_bytes=1000, max_lines=100, retain=retain))
        output.push(encoded[:split])
        output.push(encoded[split:])
        output.end()
        assert output.snapshot().text == "hello"
