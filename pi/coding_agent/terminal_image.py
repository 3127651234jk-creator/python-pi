"""Terminal inline images: capability detection, Kitty/iTerm2 encoders, cell
sizing and text fallbacks.

Port of ``packages/tui/src/terminal-image.ts``. The transcript in
``tui.TerminalUI`` is painted by prompt_toolkit, which parses ANSI escape
sequences instead of forwarding them, so Kitty/iTerm2 placements only work
when output goes straight to the terminal (``--tui off``, one-shot prompts and
the streaming editor above the prompt). The fullscreen transcript falls back to
truecolor half-block cells, which are ordinary styled text and survive every
repaint. Terminals without truecolor get the ``[Image: ...]`` text fallback.

Environment overrides match the TS implementation: ``PI_IMAGE_PROTOCOL``
(``kitty``/``iterm2``/``none``), ``PI_TRUE_COLOR`` and ``PI_HYPERLINKS``
(``1``/``0``).
"""

from __future__ import annotations

import base64
import io
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..images import convert_image_bytes_to_png

try:  # Pillow is a declared dependency; keep imports optional like images.py.
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - only reached in stripped installs
    Image = None
    ImageOps = None


KITTY_PREFIX = "\x1b_G"
ITERM2_PREFIX = "\x1b]1337;File="
BLOCK_CHAR = "\u2580"  # upper half block: fg paints the top pixel, bg the bottom


@dataclass(frozen=True)
class TerminalCapabilities:
    images: Optional[str]  # "kitty" | "iterm2" | None
    true_color: bool
    hyperlinks: bool


@dataclass(frozen=True)
class CellDimensions:
    width_px: int = 9
    height_px: int = 18


@dataclass(frozen=True)
class ImageDimensions:
    width_px: int
    height_px: int


@dataclass(frozen=True)
class ImageCellSize:
    columns: int
    rows: int


@dataclass(frozen=True)
class ImageRenderOptions:
    max_width_cells: Optional[int] = None
    max_height_cells: Optional[int] = None
    preserve_aspect_ratio: bool = True
    image_id: Optional[int] = None
    # Kitty: False emits C=1 so the cursor stays where the image was placed.
    move_cursor: bool = True


@dataclass(frozen=True)
class RenderedImage:
    sequence: str
    columns: int
    rows: int
    image_id: Optional[int] = None


_cached_capabilities: Optional[TerminalCapabilities] = None
_cell_dimensions = CellDimensions()


def get_cell_dimensions() -> CellDimensions:
    return _cell_dimensions


def set_cell_dimensions(dimensions: CellDimensions) -> None:
    global _cell_dimensions
    _cell_dimensions = dimensions


def set_capabilities(capabilities: TerminalCapabilities) -> None:
    """Override detected capabilities (tests, or an explicit user override)."""
    global _cached_capabilities
    _cached_capabilities = capabilities


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


def _probe_tmux_hyperlinks() -> bool:
    """Whether the attached tmux client forwards OSC 8 hyperlinks."""
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#{client_termfeatures}"],
            capture_output=True, text=True, timeout=0.25, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return "hyperlinks" in [feature.strip() for feature in result.stdout.split(",")]


def _parse_bool_override(value: Optional[str]) -> Optional[bool]:
    return True if value == "1" else False if value == "0" else None


def detect_capabilities(env: Optional[Dict[str, str]] = None,
                        tmux_forwards_hyperlink=None) -> TerminalCapabilities:
    """Detect image/hyperlink/truecolor support from the environment."""
    env = os.environ if env is None else env
    probe = tmux_forwards_hyperlink or _probe_tmux_hyperlinks
    term_program = env.get("TERM_PROGRAM", "").lower()
    terminal_emulator = env.get("TERMINAL_EMULATOR", "").lower()
    term = env.get("TERM", "").lower()
    color_term = env.get("COLORTERM", "").lower()
    has_true_color_hint = color_term in ("truecolor", "24bit") or term.endswith("-direct")

    # Image protocols are unreliable under tmux/screen, so keep them off there.
    if env.get("TMUX") or term.startswith("tmux"):
        detected = TerminalCapabilities(None, has_true_color_hint, probe())
    elif term.startswith("screen"):
        detected = TerminalCapabilities(None, has_true_color_hint, False)
    elif env.get("KITTY_WINDOW_ID") or term_program == "kitty":
        detected = TerminalCapabilities("kitty", True, True)
    elif term_program == "ghostty" or "ghostty" in term or env.get("GHOSTTY_RESOURCES_DIR"):
        detected = TerminalCapabilities("kitty", True, True)
    elif env.get("WEZTERM_PANE") or term_program == "wezterm":
        detected = TerminalCapabilities("kitty", True, True)
    elif term_program == "warpterminal" or env.get("WARP_SESSION_ID") or env.get("WARP_TERMINAL_SESSION_UUID"):
        detected = TerminalCapabilities("kitty", True, True)
    elif env.get("ITERM_SESSION_ID") or term_program == "iterm.app":
        detected = TerminalCapabilities("iterm2", True, True)
    elif env.get("WT_SESSION"):
        detected = TerminalCapabilities(None, True, True)
    elif term_program in ("alacritty", "vscode", "zed"):
        detected = TerminalCapabilities(None, True, True)
    elif terminal_emulator == "jetbrains-jediterm":
        detected = TerminalCapabilities(None, True, False)
    elif sys.platform == "win32":
        detected = TerminalCapabilities(None, True, False)
    else:
        detected = TerminalCapabilities(None, has_true_color_hint, False)

    image_protocol = env.get("PI_IMAGE_PROTOCOL", "").lower()
    if image_protocol in ("kitty", "iterm2"):
        detected = TerminalCapabilities(image_protocol, detected.true_color, detected.hyperlinks)
    elif image_protocol in ("none", "0"):
        detected = TerminalCapabilities(None, detected.true_color, detected.hyperlinks)

    true_color = _parse_bool_override(env.get("PI_TRUE_COLOR"))
    hyperlinks = _parse_bool_override(env.get("PI_HYPERLINKS"))
    return TerminalCapabilities(
        images=detected.images,
        true_color=detected.true_color if true_color is None else true_color,
        hyperlinks=detected.hyperlinks if hyperlinks is None else hyperlinks,
    )


def get_capabilities() -> TerminalCapabilities:
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def _decode(base64_data: str) -> Optional[bytes]:
    try:
        return base64.b64decode(base64_data)
    except (ValueError, TypeError):
        return None


def encode_kitty(base64_data: str, columns: Optional[int] = None, rows: Optional[int] = None,
                 image_id: Optional[int] = None, move_cursor: bool = True) -> str:
    chunk_size = 4096
    params = ["a=T", "f=100", "q=2"]
    if not move_cursor:
        params.append("C=1")
    if columns:
        params.append(f"c={columns}")
    if rows:
        params.append(f"r={rows}")
    if image_id:
        params.append(f"i={image_id}")

    if len(base64_data) <= chunk_size:
        return f"\x1b_G{','.join(params)};{base64_data}\x1b\\"

    chunks: List[str] = []
    offset = 0
    first = True
    while offset < len(base64_data):
        chunk = base64_data[offset:offset + chunk_size]
        is_last = offset + chunk_size >= len(base64_data)
        if first:
            chunks.append(f"\x1b_G{','.join(params)},m=1;{chunk}\x1b\\")
            first = False
        elif is_last:
            chunks.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            chunks.append(f"\x1b_Gm=1;{chunk}\x1b\\")
        offset += chunk_size
    return "".join(chunks)


def delete_kitty_image(image_id: int) -> str:
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def delete_all_kitty_images() -> str:
    return "\x1b_Ga=d,d=A,q=2\x1b\\"


def encode_iterm2(base64_data: str, width=None, height=None, name: Optional[str] = None,
                  preserve_aspect_ratio: bool = True, inline: bool = True) -> str:
    decoded = _decode(base64_data) or b""
    params = [f"inline={1 if inline else 0}", f"size={len(decoded)}"]
    if width is not None:
        params.append(f"width={width}")
    if height is not None:
        params.append(f"height={height}")
    if name:
        params.append("name=" + base64.b64encode(name.encode("utf-8")).decode("ascii"))
    if not preserve_aspect_ratio:
        params.append("preserveAspectRatio=0")
    return f"\x1b]1337;File={';'.join(params)}:{base64_data}\x07"


def get_png_dimensions(base64_data: str) -> Optional[ImageDimensions]:
    buffer = _decode(base64_data)
    if buffer is None or len(buffer) < 24:
        return None
    if buffer[0:4] != b"\x89PNG":
        return None
    return ImageDimensions(int.from_bytes(buffer[16:20], "big"), int.from_bytes(buffer[20:24], "big"))


def get_jpeg_dimensions(base64_data: str) -> Optional[ImageDimensions]:
    buffer = _decode(base64_data)
    if buffer is None or len(buffer) < 2 or buffer[0:2] != b"\xff\xd8":
        return None
    offset = 2
    while offset < len(buffer) - 9:
        if buffer[offset] != 0xFF:
            offset += 1
            continue
        marker = buffer[offset + 1]
        if 0xC0 <= marker <= 0xC2:
            height = int.from_bytes(buffer[offset + 5:offset + 7], "big")
            width = int.from_bytes(buffer[offset + 7:offset + 9], "big")
            return ImageDimensions(width, height)
        length = int.from_bytes(buffer[offset + 2:offset + 4], "big")
        if length < 2:
            return None
        offset += 2 + length
    return None


def get_gif_dimensions(base64_data: str) -> Optional[ImageDimensions]:
    buffer = _decode(base64_data)
    if buffer is None or len(buffer) < 10 or buffer[0:6] not in (b"GIF87a", b"GIF89a"):
        return None
    return ImageDimensions(int.from_bytes(buffer[6:8], "little"), int.from_bytes(buffer[8:10], "little"))


def get_webp_dimensions(base64_data: str) -> Optional[ImageDimensions]:
    buffer = _decode(base64_data)
    if buffer is None or len(buffer) < 30:
        return None
    if buffer[0:4] != b"RIFF" or buffer[8:12] != b"WEBP":
        return None
    chunk = buffer[12:16]
    if chunk == b"VP8 ":
        return ImageDimensions(int.from_bytes(buffer[26:28], "little") & 0x3FFF,
                               int.from_bytes(buffer[28:30], "little") & 0x3FFF)
    if chunk == b"VP8L":
        if len(buffer) < 25:
            return None
        bits = int.from_bytes(buffer[21:25], "little")
        return ImageDimensions((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        width = buffer[24] | (buffer[25] << 8) | (buffer[26] << 16)
        height = buffer[27] | (buffer[28] << 8) | (buffer[29] << 16)
        return ImageDimensions(width + 1, height + 1)
    return None


def get_image_dimensions(base64_data: str, mime_type: str) -> Optional[ImageDimensions]:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


def _choose_less_distorted_cell_count(upper_count: int, ideal_count: float) -> int:
    if upper_count <= 1:
        return upper_count
    lower_count = upper_count - 1
    upper_distortion = max(upper_count / ideal_count, ideal_count / upper_count)
    lower_distortion = max(lower_count / ideal_count, ideal_count / lower_count)
    return lower_count if lower_distortion < upper_distortion else upper_count


def calculate_image_cell_size(image_dimensions: ImageDimensions, max_width_cells: int,
                              max_height_cells: Optional[int] = None,
                              cell_dimensions: Optional[CellDimensions] = None,
                              optimize_aspect_ratio: bool = False) -> ImageCellSize:
    cell = cell_dimensions or get_cell_dimensions()
    max_width = max(1, int(max_width_cells))
    max_height = None if max_height_cells is None else max(1, int(max_height_cells))
    image_width = max(1, image_dimensions.width_px)
    image_height = max(1, image_dimensions.height_px)

    width_scale = (max_width * cell.width_px) / image_width
    height_scale = width_scale if max_height is None else (max_height * cell.height_px) / image_height
    scale = min(width_scale, height_scale)

    scaled_width_px = image_width * scale
    scaled_height_px = image_height * scale
    columns = max(1, min(max_width, math.ceil(scaled_width_px / cell.width_px)))
    rows = max(1, math.ceil(scaled_height_px / cell.height_px))
    if max_height is not None:
        rows = min(max_height, rows)

    if not optimize_aspect_ratio:
        return ImageCellSize(columns, rows)

    if width_scale <= height_scale:
        ideal_rows = (columns * cell.width_px * image_height) / (image_width * cell.height_px)
        rows = _choose_less_distorted_cell_count(rows, ideal_rows)
    else:
        ideal_columns = (rows * cell.height_px * image_width) / (image_height * cell.width_px)
        columns = _choose_less_distorted_cell_count(columns, ideal_columns)

    return ImageCellSize(columns, rows)


def render_image(base64_data: str, image_dimensions: ImageDimensions,
                 options: Optional[ImageRenderOptions] = None) -> Optional[RenderedImage]:
    options = options or ImageRenderOptions()
    capabilities = get_capabilities()
    if not capabilities.images:
        return None

    max_width = options.max_width_cells or 80
    size = calculate_image_cell_size(
        image_dimensions, max_width, options.max_height_cells,
        get_cell_dimensions(), capabilities.images == "kitty",
    )

    if capabilities.images == "kitty":
        sequence = encode_kitty(base64_data, columns=size.columns, rows=size.rows,
                                image_id=options.image_id, move_cursor=options.move_cursor)
        return RenderedImage(sequence, size.columns, size.rows, options.image_id)

    sequence = encode_iterm2(base64_data, width=size.columns, height="auto",
                             preserve_aspect_ratio=options.preserve_aspect_ratio)
    return RenderedImage(sequence, size.columns, size.rows)


def _shorten_image_path(filename: str) -> str:
    home = os.path.expanduser("~")
    if home and (filename == home or filename.startswith(home + os.sep)):
        return "~" + filename[len(home):]
    return filename


def hyperlink(text: str, url: str) -> str:
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def image_fallback(mime_type: str, dimensions: Optional[ImageDimensions] = None,
                   filename: Optional[str] = None,
                   capabilities: Optional[TerminalCapabilities] = None) -> str:
    capabilities = capabilities or get_capabilities()
    parts: List[str] = []
    if filename:
        display = _shorten_image_path(filename)
        uri: Optional[str] = None
        if capabilities.hyperlinks and os.path.isabs(filename):
            try:
                uri = Path(filename).as_uri()
            except ValueError:
                uri = None
        parts.append(hyperlink(display, uri) if uri else display)
    parts.append(f"[{mime_type}]")
    if dimensions is not None:
        parts.append(f"{dimensions.width_px}x{dimensions.height_px}")
    return f"[Image: {' '.join(parts)}]"


def render_block_art(base64_data: str, max_width_cells: int,
                     max_height_cells: Optional[int] = None) -> Optional[List[str]]:
    """Render an image as truecolor half-block text (two pixels per cell)."""
    if Image is None or ImageOps is None:
        return None
    data = _decode(base64_data)
    if data is None:
        return None
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGB", rgba.size, (0, 0, 0))
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
    except Exception:  # noqa: BLE001 - undecodable images fall back to text
        return None

    width_cells = max(1, int(max_width_cells))
    scale = width_cells / image.width
    if max_height_cells is not None:
        scale = min(scale, (max(1, int(max_height_cells)) * 2) / image.height)
    target_width = max(1, min(width_cells, round(image.width * scale)))
    target_height = max(2, round(image.height * scale))
    if target_height % 2:
        target_height += 1
    if (target_width, target_height) != image.size:
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    pixels = image.load()
    lines: List[str] = []
    for y in range(0, target_height, 2):
        parts: List[str] = []
        for x in range(target_width):
            top = pixels[x, y]
            bottom = pixels[x, y + 1]
            parts.append(
                f"\x1b[38;2;{top[0]};{top[1]};{top[2]}m"
                f"\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m{BLOCK_CHAR}"
            )
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return lines


def render_image_lines(base64_data: str, mime_type: str, filename: Optional[str] = None,
                       max_width_cells: int = 60, max_height_cells: Optional[int] = None,
                       allow_protocol: bool = True) -> List[str]:
    """Render one image into terminal lines.

    With ``allow_protocol`` and a supported terminal the result contains a
    Kitty/iTerm2 placement plus blank rows that reserve its height. Otherwise
    it contains truecolor half-block lines, or the text fallback.
    """
    capabilities = get_capabilities()
    dimensions = get_image_dimensions(base64_data, mime_type)

    # Bound height by width so a very tall image cannot flood the transcript.
    height_limit = max_height_cells
    if height_limit is None:
        cell = get_cell_dimensions()
        height_limit = max(1, math.ceil(max_width_cells * cell.width_px / cell.height_px))

    if allow_protocol and capabilities.images:
        data: Optional[str] = base64_data
        if capabilities.images == "kitty" and mime_type != "image/png":
            # Kitty only accepts PNG payloads.
            converted = convert_image_bytes_to_png(_decode(base64_data) or b"")
            if converted is not None:
                data = base64.b64encode(converted).decode("ascii")
                dimensions = get_png_dimensions(data) or dimensions
            else:
                data = None
        if data is not None:
            rendered = render_image(
                data, dimensions or ImageDimensions(800, 600),
                ImageRenderOptions(max_width_cells=max_width_cells,
                                   max_height_cells=height_limit,
                                   move_cursor=False),
            )
            if rendered is not None:
                if capabilities.images == "kitty":
                    # C=1 leaves the cursor in place; the blank rows reserve the
                    # rows the placement covers.
                    return [rendered.sequence] + [""] * (rendered.rows - 1)
                # iTerm2 draws at the cursor and advances past the image.
                row_offset = rendered.rows - 1
                move_up = f"\x1b[{row_offset}A" if row_offset > 0 else ""
                return [""] * row_offset + [move_up + rendered.sequence]

    if capabilities.true_color:
        lines = render_block_art(base64_data, max_width_cells, height_limit)
        if lines:
            return lines

    return [image_fallback(mime_type, dimensions, filename, capabilities)]
