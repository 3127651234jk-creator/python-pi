"""Inline-image pipeline: MIME sniffing, PNG conversion, model-fit resizing.

Shared by the coding-agent tools and the durable tools. Port of
``packages/coding-agent/src/utils/{mime,image-convert,image-resize-core,
image-process}.ts``. MIME detection is magic-byte based because extensions lie;
formats the providers do not accept inline are converted to PNG with Pillow,
the stand-in for the TS Photon WASM module. If Pillow is missing, conversion
and resizing fail gracefully and callers get an "[Image omitted: ...]" note,
exactly like a failed Photon load in TS.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# MIME detection (port of mime.ts)
# ---------------------------------------------------------------------------

IMAGE_TYPE_SNIFF_BYTES = 4100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def detect_supported_image_mime_type(buffer: bytes) -> Optional[str]:
    """Detect an inline-supported image MIME type from leading magic bytes."""
    if buffer.startswith(b"\xff\xd8\xff"):
        return None if buffer[3:4] == b"\xf7" else "image/jpeg"
    if buffer.startswith(PNG_SIGNATURE):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if buffer.startswith(b"GIF87a") or buffer.startswith(b"GIF89a"):
        return "image/gif"
    if buffer.startswith(b"RIFF") and buffer[8:12] == b"WEBP":
        return "image/webp"
    if buffer.startswith(b"BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def detect_supported_image_mime_type_from_file(path: str) -> Optional[str]:
    """Sniff the first ``IMAGE_TYPE_SNIFF_BYTES`` of a file for an image type."""
    with open(path, "rb") as handle:
        buffer = handle.read(IMAGE_TYPE_SNIFF_BYTES)
    return detect_supported_image_mime_type(buffer)


def _is_png(buffer: bytes) -> bool:
    return len(buffer) >= 16 and _read_uint32_be(buffer, 8) == 13 and buffer[12:16] == b"IHDR"


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _read_uint32_be(buffer, offset)
        chunk_type = buffer[offset + 4 : offset + 8]
        if chunk_type == b"acTL":
            return True
        if chunk_type == b"IDAT":
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_file_size = _read_uint32_le(buffer, 2)
    pixel_data_offset = _read_uint32_le(buffer, 10)
    dib_header_size = _read_uint32_le(buffer, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False
    if dib_header_size == 12:
        color_planes = _read_uint16_le(buffer, 22)
        bits_per_pixel = _read_uint16_le(buffer, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buffer) < 30:
            return False
        color_planes = _read_uint16_le(buffer, 26)
        bits_per_pixel = _read_uint16_le(buffer, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _read_uint16_le(buffer: bytes, offset: int) -> int:
    return buffer[offset] | (buffer[offset + 1] << 8)


def _read_uint32_be(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 4], "big")


def _read_uint32_le(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 4], "little")


# ---------------------------------------------------------------------------
# PNG conversion (port of image-convert.ts)
# ---------------------------------------------------------------------------


def convert_image_bytes_to_png(data: bytes) -> Optional[bytes]:
    """Decode any Pillow-supported image and re-encode as PNG; None on failure."""
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            out = io.BytesIO()
            oriented.save(out, format="PNG")
            return out.getvalue()
    except Exception:  # noqa: BLE001 - failed conversion is a null result, like TS
        return None


# ---------------------------------------------------------------------------
# Resizing (port of image-resize-core.ts)
# ---------------------------------------------------------------------------

# 4.5MB of base64 payload. Provides headroom below Anthropic's 5MB limit.
DEFAULT_MAX_BYTES = 4_718_592


@dataclass
class ImageResizeOptions:
    max_width: int = 2000
    max_height: int = 2000
    max_bytes: int = DEFAULT_MAX_BYTES
    jpeg_quality: int = 80


@dataclass
class ResizedImage:
    data: str  # base64
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


@dataclass
class _EncodedCandidate:
    data: str
    mime_type: str


def resize_options_from_profile(profile: Any) -> ImageResizeOptions:
    """Merge an all-optional ``ModelImageResizeOptions`` profile onto the defaults."""
    options = ImageResizeOptions()
    if profile is None:
        return options
    return ImageResizeOptions(
        max_width=profile.max_width if profile.max_width is not None else options.max_width,
        max_height=profile.max_height if profile.max_height is not None else options.max_height,
        max_bytes=profile.max_bytes if profile.max_bytes is not None else options.max_bytes,
        jpeg_quality=profile.jpeg_quality if profile.jpeg_quality is not None else options.jpeg_quality,
    )


def _load_image(data: bytes) -> Optional[Any]:
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    try:
        image = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(image)
    except Exception:  # noqa: BLE001 - undecodable input is a null result
        return None


def resize_image(input_bytes: bytes, mime_type: str, options: Optional[ImageResizeOptions] = None) -> Optional[ResizedImage]:
    """Resize an image to fit within the configured dimensions and encoded size.

    Strategy for staying under ``max_bytes``: resize to the max dimensions,
    then try PNG and JPEG candidates (decreasing JPEG quality), then shrink
    dimensions by 25% at a time until 1x1. Returns None when nothing fits or
    the image cannot be decoded.
    """
    opts = options or ImageResizeOptions()
    # _load_image returns None when Pillow is unavailable or decoding fails.
    image = _load_image(input_bytes)
    if image is None:
        return None
    from PIL import Image  # noqa: PLC0415 - optional dependency, present if _load_image succeeded

    try:
        original_width, original_height = image.size
        input_base64_size = ((len(input_bytes) + 2) // 3) * 4

        # Already within all limits (dimensions AND encoded size).
        if (
            original_width <= opts.max_width
            and original_height <= opts.max_height
            and input_base64_size < opts.max_bytes
        ):
            return ResizedImage(
                data=base64.b64encode(input_bytes).decode("ascii"),
                mime_type=mime_type or "image/png",
                original_width=original_width,
                original_height=original_height,
                width=original_width,
                height=original_height,
                was_resized=False,
            )

        target_width, target_height = original_width, original_height
        if target_width > opts.max_width:
            target_height = max(1, int(target_height * opts.max_width / target_width + 0.5))
            target_width = opts.max_width
        if target_height > opts.max_height:
            target_width = max(1, int(target_width * opts.max_height / target_height + 0.5))
            target_height = opts.max_height

        try:
            resampling = Image.Resampling.LANCZOS
        except AttributeError:  # Pillow < 9.1
            resampling = Image.LANCZOS

        quality_steps = list(dict.fromkeys([opts.jpeg_quality, 85, 70, 55, 40]))
        current_width, current_height = target_width, target_height

        while True:
            candidates: List[_EncodedCandidate] = []
            resized = image.resize((current_width, current_height), resampling)
            png_buffer = io.BytesIO()
            resized.save(png_buffer, format="PNG")
            candidates.append(_EncodedCandidate(base64.b64encode(png_buffer.getvalue()).decode("ascii"), "image/png"))
            # JPEG encoding rejects alpha and palette modes.
            jpeg_source = resized if resized.mode in ("RGB", "L") else resized.convert("RGB")
            for quality in quality_steps:
                jpeg_buffer = io.BytesIO()
                jpeg_source.save(jpeg_buffer, format="JPEG", quality=quality)
                candidates.append(
                    _EncodedCandidate(base64.b64encode(jpeg_buffer.getvalue()).decode("ascii"), "image/jpeg")
                )

            for candidate in candidates:
                if len(candidate.data) < opts.max_bytes:
                    return ResizedImage(
                        data=candidate.data,
                        mime_type=candidate.mime_type,
                        original_width=original_width,
                        original_height=original_height,
                        width=current_width,
                        height=current_height,
                        was_resized=True,
                    )

            if current_width == 1 and current_height == 1:
                return None

            next_width = 1 if current_width == 1 else max(1, int(current_width * 0.75))
            next_height = 1 if current_height == 1 else max(1, int(current_height * 0.75))
            if next_width == current_width and next_height == current_height:
                return None
            current_width, current_height = next_width, next_height
    except Exception:  # noqa: BLE001 - failed resizing is a null result, like TS
        return None


def format_dimension_note(result: ResizedImage) -> Optional[str]:
    """Note for resized images so the model can map coordinates back."""
    if not result.was_resized:
        return None
    scale = result.original_width / result.width
    return (
        f"[Image: original {result.original_width}x{result.original_height}, "
        f"displayed at {result.width}x{result.height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


# ---------------------------------------------------------------------------
# Processing (port of image-process.ts)
# ---------------------------------------------------------------------------


@dataclass
class ProcessImageOptions:
    auto_resize_images: bool = True
    resize_options: Optional[ImageResizeOptions] = None


@dataclass
class ProcessedImage:
    ok: bool
    data: str = ""  # base64, when ok
    mime_type: str = ""
    hints: List[str] = field(default_factory=list)
    message: str = ""  # failure note, when not ok


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";", 1)[0].strip().lower()


def _normalize_supported_image_mime_type(mime_type: str) -> Optional[str]:
    return {
        "image/png": "image/png",
        "image/jpeg": "image/jpeg",
        "image/jpg": "image/jpeg",
        "image/gif": "image/gif",
        "image/webp": "image/webp",
    }.get(_base_mime_type(mime_type))


def _normalize_image_sync(data: bytes, mime_type: str) -> Optional[Tuple[bytes, str, Optional[str]]]:
    normalized_mime = _normalize_supported_image_mime_type(mime_type)
    if normalized_mime:
        return data, normalized_mime, None
    png_bytes = convert_image_bytes_to_png(data)
    if png_bytes is None:
        return None
    return png_bytes, "image/png", _base_mime_type(mime_type)


def _conversion_hint(from_mime: Optional[str], to_mime: str) -> Optional[str]:
    if not from_mime or from_mime == to_mime:
        return None
    return f"[Image converted from {from_mime} to {to_mime}.]"


def _process_image_sync(data: bytes, mime_type: str, options: ProcessImageOptions) -> ProcessedImage:
    normalized = _normalize_image_sync(data, mime_type)
    if normalized is None:
        return ProcessedImage(
            ok=False, message="[Image omitted: could not be converted to a supported inline image format.]"
        )
    image_bytes, normalized_mime, converted_from = normalized

    if options.auto_resize_images:
        resized = resize_image(image_bytes, normalized_mime, options.resize_options)
        if resized is None:
            return ProcessedImage(
                ok=False, message="[Image omitted: could not be resized below the inline image size limit.]"
            )
        hints: List[str] = []
        converted_hint = _conversion_hint(converted_from, resized.mime_type)
        if converted_hint:
            hints.append(converted_hint)
        dimension_note = format_dimension_note(resized)
        if dimension_note:
            hints.append(dimension_note)
        return ProcessedImage(ok=True, data=resized.data, mime_type=resized.mime_type, hints=hints)

    hints = []
    converted_hint = _conversion_hint(converted_from, normalized_mime)
    if converted_hint:
        hints.append(converted_hint)
    return ProcessedImage(
        ok=True, data=base64.b64encode(image_bytes).decode("ascii"), mime_type=normalized_mime, hints=hints
    )


async def process_image(
    data: bytes, mime_type: str, options: Optional[ProcessImageOptions] = None
) -> ProcessedImage:
    """Normalize and resize one image for inline provider transport.

    Pillow decoding and encoding run in a worker thread so they cannot block
    the event loop, matching the TS worker-thread offload.
    """
    import asyncio  # noqa: PLC0415 - kept local to avoid import cost at module load

    resolved = options or ProcessImageOptions()
    return await asyncio.to_thread(_process_image_sync, data, mime_type, resolved)
