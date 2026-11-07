"""Image pipeline tests: MIME detection, PNG conversion, resizing, blockImages,
@file processing, and openrouter-images request/response handling.

Ports of packages/coding-agent/test/{image-process,block-images}.test.ts plus
packages/ai/test/openrouter-images.test.ts coverage.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from PIL import Image

from pi.agent.agent_loop import IMAGE_DISABLED_TEXT, default_convert_to_llm
from pi.ai.models import Model, ModelCost, parse_input_limits
from pi.ai.providers.openrouter_images import build_params, parse_image_response
from pi.ai.types import AssistantImages, ImageContent, ImagesContext, TextContent, ToolResultMessage, UserMessage
from pi.coding_agent.file_processor import process_file_arguments
from pi.images import (
    ImageResizeOptions,
    ProcessImageOptions,
    detect_supported_image_mime_type,
    format_dimension_note,
    process_image,
    resize_image,
)
from pi.coding_agent.tools import create_default_tools

# 1x1 red PNG image as base64 (smallest valid PNG)
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)

PNG_MAGIC = b"\x89PNG"


def create_tiny_bmp_1x1_red_24bpp() -> bytes:
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    struct.pack_into("<I", buffer, 2, len(buffer))
    struct.pack_into("<I", buffer, 10, 54)
    struct.pack_into("<I", buffer, 14, 40)
    struct.pack_into("<i", buffer, 18, 1)
    struct.pack_into("<i", buffer, 22, 1)
    struct.pack_into("<H", buffer, 26, 1)
    struct.pack_into("<H", buffer, 28, 24)
    struct.pack_into("<I", buffer, 30, 0)
    struct.pack_into("<I", buffer, 34, 4)
    buffer[56] = 0xFF
    return bytes(buffer)


def create_apng_header() -> bytes:
    """PNG signature + IHDR + acTL + IDAT: an animated PNG by chunk layout."""

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + ctype + data + b"\x00\x00\x00\x00"

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    actl = struct.pack(">II", 1, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"acTL", actl) + chunk(b"IDAT", b"\x00")


def large_png(width: int = 3000, height: int = 2000) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def image_model(output=None) -> Model:
    return Model(
        id="google/gemini-2.5-flash-image",
        name="Gemini Image",
        api="openrouter-images",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        output=output or ["image", "text"],
        costs=ModelCost(input=1.0, output=2.0),
    )


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------


def test_detects_gif_signatures():
    assert detect_supported_image_mime_type(b"GIF87a") == "image/gif"
    assert detect_supported_image_mime_type(b"GIF89a") == "image/gif"


def test_detects_bmp_from_magic_bytes():
    assert detect_supported_image_mime_type(create_tiny_bmp_1x1_red_24bpp()) == "image/bmp"


def test_detects_jpeg_png_webp_and_rejects_apng():
    assert detect_supported_image_mime_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    # 0xf7 in the fourth byte is excluded, like TS.
    assert detect_supported_image_mime_type(b"\xff\xd8\xff\xf7rest") is None
    assert detect_supported_image_mime_type(base64.b64decode(TINY_PNG_BASE64)) == "image/png"
    assert detect_supported_image_mime_type(create_apng_header()) is None
    assert detect_supported_image_mime_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert detect_supported_image_mime_type(b"not an image") is None


# ---------------------------------------------------------------------------
# process_image: conversion + resizing
# ---------------------------------------------------------------------------


def test_process_image_converts_bmp_without_resize():
    result = asyncio.run(
        process_image(create_tiny_bmp_1x1_red_24bpp(), "image/bmp", ProcessImageOptions(auto_resize_images=False))
    )
    assert result.ok
    assert result.mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.hints
    assert base64.b64decode(result.data)[:4] == PNG_MAGIC


def test_process_image_converts_bmp_before_resizing():
    result = asyncio.run(process_image(create_tiny_bmp_1x1_red_24bpp(), "image/bmp"))
    assert result.ok
    assert result.mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.hints


def test_process_image_resizes_to_model_limits():
    result = asyncio.run(process_image(large_png(), "image/png"))
    assert result.ok
    assert result.mime_type in ("image/png", "image/jpeg")
    assert any("displayed at 2000x1333" in hint for hint in result.hints)
    assert any("Multiply coordinates by 1.50" in hint for hint in result.hints)


def test_process_image_reports_unconvertible_input():
    result = asyncio.run(process_image(b"not an image", "image/x-unknown"))
    assert not result.ok
    assert result.message == "[Image omitted: could not be converted to a supported inline image format.]"


def test_resize_image_dimension_note():
    result = resize_image(large_png(), "image/png")
    assert result is not None
    assert result.was_resized
    assert result.width == 2000 and result.height == 1333
    assert result.original_width == 3000 and result.original_height == 2000
    expected = (
        "[Image: original 3000x2000, displayed at 2000x1333. "
        "Multiply coordinates by 1.50 to map to original image.]"
    )
    assert format_dimension_note(result) == expected


def test_resize_image_shrinks_until_under_max_bytes():
    result = resize_image(large_png(), "image/png", ImageResizeOptions(max_bytes=600))
    assert result is not None
    assert result.was_resized
    assert len(result.data) < 600


def test_resize_image_passthrough_within_limits():
    data = base64.b64decode(TINY_PNG_BASE64)
    result = resize_image(data, "image/png")
    assert result is not None
    assert not result.was_resized
    assert result.data == TINY_PNG_BASE64
    assert format_dimension_note(result) is None


# ---------------------------------------------------------------------------
# CLI @file attachments (port of block-images.test.ts processFileArguments)
# ---------------------------------------------------------------------------


def test_process_file_arguments(tmp_path):
    (tmp_path / "test.png").write_bytes(base64.b64decode(TINY_PNG_BASE64))
    (tmp_path / "test.bmp").write_bytes(create_tiny_bmp_1x1_red_24bpp())
    (tmp_path / "test.txt").write_text("Hello, world!", encoding="utf-8")

    result = asyncio.run(
        process_file_arguments(["test.png", "test.bmp", "test.txt"], cwd=str(tmp_path), auto_resize_images=False)
    )
    assert len(result.images) == 2
    assert result.images[0].mime_type == "image/png"
    assert result.images[1].mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.text
    assert "Hello, world!" in result.text
    assert f'<file name="{tmp_path / "test.png"}"></file>' in result.text


def test_process_file_arguments_missing_file(tmp_path):
    try:
        asyncio.run(process_file_arguments(["nope.png"], cwd=str(tmp_path)))
        raised = None
    except FileNotFoundError as exc:
        raised = exc
    assert raised is not None and "File not found" in str(raised)


# ---------------------------------------------------------------------------
# Read tool: images always read, filtering happens at convert_to_llm
# ---------------------------------------------------------------------------


def test_read_tool_reads_image_content(tmp_path):
    (tmp_path / "test.png").write_bytes(base64.b64decode(TINY_PNG_BASE64))
    tool = next(t for t in create_default_tools(str(tmp_path)) if t.name == "read")

    result = asyncio.run(tool.execute("test-1", {"path": str(tmp_path / "test.png")}))
    types = [block.type for block in result.content]
    assert "image" in types
    assert result.content[0].text.startswith("Read image file [image/png]")


def test_read_tool_text_files(tmp_path):
    (tmp_path / "test.txt").write_text("Hello, world!", encoding="utf-8")
    tool = next(t for t in create_default_tools(str(tmp_path)) if t.name == "read")

    result = asyncio.run(tool.execute("test-2", {"path": str(tmp_path / "test.txt")}))
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "Hello, world!" in result.content[0].text


def test_read_tool_non_vision_note(tmp_path):
    (tmp_path / "test.png").write_bytes(base64.b64decode(TINY_PNG_BASE64))
    model = Model(id="m", name="m", api="faux", provider="faux", base_url="", input=["text"])
    tool = next(t for t in create_default_tools(str(tmp_path), get_model=lambda: model) if t.name == "read")

    result = asyncio.run(tool.execute("t", {"path": str(tmp_path / "test.png")}))
    assert "does not support images" in result.content[0].text


# ---------------------------------------------------------------------------
# blockImages: filtering at the convert_to_llm boundary
# ---------------------------------------------------------------------------


def test_convert_to_llm_blocks_images_with_dedupe():
    user = UserMessage(
        content=[
            TextContent(text="look"),
            ImageContent(data="x", mime_type="image/png"),
            ImageContent(data="y", mime_type="image/png"),
        ],
        timestamp=0,
    )
    converted = default_convert_to_llm([user], block_images=True)
    assert [block.text for block in converted[0].content] == ["look", IMAGE_DISABLED_TEXT]


def test_convert_to_llm_blocks_tool_result_images():
    tool_result = ToolResultMessage(
        tool_call_id="1",
        tool_name="read",
        content=[ImageContent(data="x", mime_type="image/png")],
        is_error=False,
        timestamp=0,
    )
    converted = default_convert_to_llm([tool_result], block_images=True)
    assert [block.text for block in converted[0].content] == [IMAGE_DISABLED_TEXT]


def test_convert_to_llm_keeps_images_by_default():
    user = UserMessage(content=[ImageContent(data="x", mime_type="image/png")], timestamp=0)
    converted = default_convert_to_llm([user])
    assert converted[0].content[0].type == "image"


# ---------------------------------------------------------------------------
# openrouter-images
# ---------------------------------------------------------------------------


def test_build_params_modalities_follow_model_output():
    model = image_model()
    params = build_params(
        model,
        ImagesContext(input=[TextContent(text="a cat"), ImageContent(data="QUJD", mime_type="image/png")]),
    )
    assert params["model"] == model.id
    assert params["stream"] is False
    assert params["modalities"] == ["image", "text"]
    assert params["messages"][0]["content"][0] == {"type": "text", "text": "a cat"}
    assert params["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    image_only = build_params(replace(model, output=["image"]), ImagesContext())
    assert image_only["modalities"] == ["image"]


def test_parse_image_response_text_and_images():
    payload = {
        "id": "resp-1",
        "choices": [
            {
                "message": {
                    "content": "here you go",
                    "images": [
                        {"image_url": {"url": "data:image/png;base64,QUJD"}},
                        {"image_url": "https://example.com/remote.png"},
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    model = image_model()
    output = AssistantImages(api=model.api, provider=model.provider, model=model.id, timestamp=0)
    parse_image_response(payload, model, output)
    assert output.response_id == "resp-1"
    assert output.output == [
        TextContent(text="here you go"),
        ImageContent(data="QUJD", mime_type="image/png"),
    ]
    assert output.usage is not None
    assert output.usage.input == 10 and output.usage.output == 5
    assert output.usage.cost["total"] == (10 * 1.0 + 5 * 2.0) / 1_000_000


def test_parse_input_limits_profile():
    limits = parse_input_limits({"images": {"resize": {"maxWidth": 1024, "jpegQuality": 75}, "maxPerMessage": 5}})
    assert limits is not None and limits.images is not None
    assert limits.images.resize is not None
    assert limits.images.resize.max_width == 1024
    assert limits.images.resize.jpeg_quality == 75
    assert limits.images.max_per_message == 5
    assert limits.images.resize.max_height is None
    assert parse_input_limits(None) is None
