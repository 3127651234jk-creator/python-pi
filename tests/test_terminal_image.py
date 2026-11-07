import asyncio
import base64
import io

import pytest
from PIL import Image
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pi.agent.agent import Agent
from pi.agent.types import AgentToolResult, ToolExecutionEndEvent
from pi.ai.models import resolve_model
from pi.ai.types import ImageContent, TextContent
from pi.coding_agent import terminal_image as ti
from pi.coding_agent.cli import CliRenderer
from pi.coding_agent.tui import TerminalUI


def encode_image(fmt: str, size, color=(200, 30, 40)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture(autouse=True)
def reset_terminal_image():
    ti.reset_capabilities_cache()
    ti.set_cell_dimensions(ti.CellDimensions())
    yield
    ti.reset_capabilities_cache()
    ti.set_cell_dimensions(ti.CellDimensions())


def test_detect_capabilities_for_known_terminals():
    assert ti.detect_capabilities({"KITTY_WINDOW_ID": "1"}) == ti.TerminalCapabilities("kitty", True, True)
    assert ti.detect_capabilities({"ITERM_SESSION_ID": "abc"}).images == "iterm2"
    assert ti.detect_capabilities({"WEZTERM_PANE": "0"}).images == "kitty"
    tmux = ti.detect_capabilities({"TMUX": "1", "TERM": "tmux-256color"}, lambda: True)
    assert tmux.images is None and tmux.hyperlinks is True


def test_detect_capabilities_honors_overrides():
    caps = ti.detect_capabilities(
        {"PI_IMAGE_PROTOCOL": "iterm2", "PI_TRUE_COLOR": "0", "PI_HYPERLINKS": "0"})
    assert caps == ti.TerminalCapabilities("iterm2", False, False)
    assert ti.detect_capabilities({"PI_IMAGE_PROTOCOL": "none"}).images is None


def test_encode_kitty_chunks_long_payloads():
    sequence = ti.encode_kitty("A" * 9000, columns=10, rows=5, image_id=7, move_cursor=False)
    assert sequence.startswith("\x1b_Ga=T,f=100,q=2,C=1,c=10,r=5,i=7,m=1;")
    assert sequence.count("\x1b_G") == 3
    assert "\x1b_Gm=1;" in sequence
    assert sequence.endswith("\x1b_Gm=0;" + "A" * 808 + "\x1b\\")


def test_encode_kitty_single_chunk_and_delete():
    assert ti.encode_kitty("QUJD", columns=3) == "\x1b_Ga=T,f=100,q=2,c=3;QUJD\x1b\\"
    assert ti.delete_kitty_image(5) == "\x1b_Ga=d,d=I,i=5,q=2\x1b\\"


def test_encode_iterm2_controls():
    payload = encode_image("PNG", (4, 4))
    sequence = ti.encode_iterm2(payload, width=4, height="auto", name="shot.png",
                                preserve_aspect_ratio=False)
    assert sequence.startswith("\x1b]1337;File=inline=1;size=")
    assert ";width=4;height=auto;name=c2hvdC5wbmc=;preserveAspectRatio=0:" in sequence
    assert sequence.endswith("\x07")


def test_get_image_dimensions_across_formats():
    assert ti.get_image_dimensions(encode_image("PNG", (13, 7)), "image/png") == ti.ImageDimensions(13, 7)
    assert ti.get_image_dimensions(encode_image("JPEG", (11, 9)), "image/jpeg") == ti.ImageDimensions(11, 9)
    assert ti.get_image_dimensions(encode_image("GIF", (5, 3)), "image/gif") == ti.ImageDimensions(5, 3)
    assert ti.get_image_dimensions("not an image", "image/png") is None


def test_calculate_image_cell_size_respects_bounds():
    dimensions = ti.ImageDimensions(90, 180)
    assert ti.calculate_image_cell_size(dimensions, 10) == ti.ImageCellSize(10, 10)
    assert ti.calculate_image_cell_size(dimensions, 10, max_height_cells=5) == ti.ImageCellSize(5, 5)


def test_calculate_image_cell_size_optimizes_kitty_distortion():
    size = ti.calculate_image_cell_size(ti.ImageDimensions(200, 88), 10, optimize_aspect_ratio=True)
    assert size == ti.ImageCellSize(10, 2)


def test_render_image_requires_protocol_support():
    ti.set_capabilities(ti.TerminalCapabilities(None, True, False))
    assert ti.render_image(encode_image("PNG", (9, 9)), ti.ImageDimensions(9, 9)) is None


def test_render_image_uses_protocol_and_cell_size():
    ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
    rendered = ti.render_image(encode_image("PNG", (90, 180)), ti.ImageDimensions(90, 180),
                               ti.ImageRenderOptions(max_width_cells=10, move_cursor=False))
    assert rendered is not None
    assert (rendered.columns, rendered.rows) == (10, 10)
    assert rendered.sequence.startswith("\x1b_Ga=T,f=100,q=2,C=1,c=10,r=10;")

    ti.set_capabilities(ti.TerminalCapabilities("iterm2", True, False))
    rendered = ti.render_image(encode_image("PNG", (90, 180)), ti.ImageDimensions(90, 180),
                               ti.ImageRenderOptions(max_width_cells=10))
    assert rendered is not None
    assert rendered.sequence.startswith("\x1b]1337;File=inline=1;size=")
    assert (rendered.columns, rendered.rows) == (10, 10)


def test_protocol_lines_reserve_image_rows():
    ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
    lines = ti.render_image_lines(encode_image("PNG", (90, 180)), "image/png", max_width_cells=10)
    assert len(lines) == 5
    assert lines[0].startswith("\x1b_Ga=T,f=100,q=2,C=1,c=5,r=5;")
    assert lines[1:] == [""] * 4

    ti.set_capabilities(ti.TerminalCapabilities("iterm2", True, False))
    lines = ti.render_image_lines(encode_image("PNG", (90, 180)), "image/png", max_width_cells=10)
    assert len(lines) == 5
    assert lines[:4] == [""] * 4
    assert lines[4].startswith("\x1b[4A\x1b]1337;File=inline=1;size=")
    assert ";width=5;height=auto:" in lines[4]


def test_kitty_converts_non_png_payloads():
    ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
    lines = ti.render_image_lines(encode_image("JPEG", (20, 20)), "image/jpeg", max_width_cells=5)
    assert lines[0].startswith("\x1b_G")


def test_lines_disable_protocol_for_fullscreen_transcript():
    ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
    lines = ti.render_image_lines(encode_image("PNG", (4, 4)), "image/png",
                                  max_width_cells=2, allow_protocol=False)
    assert len(lines) == 1
    assert lines[0].startswith("\x1b[38;2;") and ti.BLOCK_CHAR in lines[0]


def test_text_fallback_without_truecolor():
    ti.set_capabilities(ti.TerminalCapabilities(None, False, False))
    assert ti.render_image_lines(encode_image("PNG", (4, 4)), "image/png", max_width_cells=2) == [
        "[Image: [image/png] 4x4]"]


def test_image_fallback_includes_filename_and_hyperlink(tmp_path):
    caps = ti.TerminalCapabilities(None, True, True)
    filename = str(tmp_path / "shot.png")
    text = ti.image_fallback("image/png", ti.ImageDimensions(4, 4), filename, caps)
    assert text.startswith("[Image: \x1b]8;;file://")
    assert "shot.png" in text and "[image/png] 4x4" in text


def test_cli_renderer_prints_tool_result_images(capsys):
    ti.set_capabilities(ti.TerminalCapabilities(None, True, False))
    renderer = CliRenderer(settings={"fold_output": 10})
    png = encode_image("PNG", (4, 4))
    result = AgentToolResult(content=[TextContent(text="Read image file [image/png]"),
                                      ImageContent(data=png, mime_type="image/png")])
    asyncio.run(renderer(ToolExecutionEndEvent(tool_call_id="1", tool_name="read", result=result)))
    out = capsys.readouterr().out
    assert "Read image file [image/png]" in out
    assert ti.BLOCK_CHAR in out
    assert "\x1b_G" not in out


def test_cli_renderer_uses_protocol_outside_fullscreen_only(capsys):
    ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
    renderer = CliRenderer(settings={})
    png = encode_image("PNG", (18, 18))

    renderer.print_image_block(ImageContent(data=png, mime_type="image/png"))
    assert "\x1b_Ga=T" in capsys.readouterr().out

    renderer.full_screen = True
    renderer.print_image_block(ImageContent(data=png, mime_type="image/png"))
    out = capsys.readouterr().out
    assert "\x1b_G" not in out and ti.BLOCK_CHAR in out


def test_fullscreen_transcript_renders_image_as_block_art():
    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe,
                        output=DummyOutput(), full_screen=True)
        ti.set_capabilities(ti.TerminalCapabilities("kitty", True, False))
        renderer = CliRenderer(settings={})
        renderer.ui = ui
        renderer.full_screen = True
        block = ImageContent(data=encode_image("PNG", (4, 4)), mime_type="image/png")
        ui.write(renderer.render_image_block(block) + "\n")

        fragments = ui.formatted_transcript()
        texts = "".join(text for _style, text in fragments)
        assert ti.BLOCK_CHAR in texts
        assert "\x1b_G" not in texts
        assert any("bg:#" in style for style, _text in fragments)

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))
