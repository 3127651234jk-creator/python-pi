"""Image MIME detection (port of ``tools/image.ts``).

``detect_supported_image_mime_type`` of whole bytes, and of a file read only at
its header and, for PNG, the chunk headers up to the first ``acTL`` or ``IDAT``.
"""

from __future__ import annotations

from typing import Awaitable, Optional

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
# Bytes every check except the APNG chunk walk needs: BMP reads up to offset 29.
HEADER_BYTES = 32
BLOCK_BYTES = 64 * 1024


class ByteSource:
    """Positional reads of a file of ``size`` bytes."""

    size: int

    def read(self, offset: int, length: int) -> Awaitable[bytes]:
        raise NotImplementedError


def starts_with(data: bytes, prefix: bytes) -> bool:
    return data[: len(prefix)] == prefix


def starts_with_ascii(data: bytes, offset: int, text: str) -> bool:
    return data[offset : offset + len(text)] == text.encode("ascii")


def read_uint16_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def read_uint32_be(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big")


def read_uint32_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def is_png(data: bytes) -> bool:
    return len(data) >= 16 and read_uint32_be(data, len(PNG_SIGNATURE)) == 13 and starts_with_ascii(data, 12, "IHDR")


def is_animated_png(data: bytes) -> bool:
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        chunk_length = read_uint32_be(data, offset)
        chunk_type_offset = offset + 4
        if starts_with_ascii(data, chunk_type_offset, "acTL"):
            return True
        if starts_with_ascii(data, chunk_type_offset, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(data):
            return False
        offset = next_offset
    return False


def is_bmp(data: bytes) -> bool:
    if len(data) < 26:
        return False
    declared_file_size = read_uint32_le(data, 2)
    pixel_data_offset = read_uint32_le(data, 10)
    dib_header_size = read_uint32_le(data, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        if len(data) < 26:
            return False
        color_planes = read_uint16_le(data, 22)
        bits_per_pixel = read_uint16_le(data, 24)
    elif 40 <= dib_header_size <= 124:
        if len(data) < 30:
            return False
        color_planes = read_uint16_le(data, 26)
        bits_per_pixel = read_uint16_le(data, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def detect_supported_image_mime_type(buffer: bytes) -> Optional[str]:
    """Supported image MIME type of the bytes, or ``None`` (animated PNG counts as unsupported)."""
    if starts_with(buffer, b"\xff\xd8\xff"):
        return None if buffer[3] == 0xF7 else "image/jpeg"
    if starts_with(buffer, PNG_SIGNATURE):
        return "image/png" if is_png(buffer) and not is_animated_png(buffer) else None
    if starts_with_ascii(buffer, 0, "GIF87a") or starts_with_ascii(buffer, 0, "GIF89a"):
        return "image/gif"
    if starts_with_ascii(buffer, 0, "RIFF") and starts_with_ascii(buffer, 8, "WEBP"):
        return "image/webp"
    if starts_with_ascii(buffer, 0, "BM") and is_bmp(buffer):
        return "image/bmp"
    return None


async def _is_animated_png_of(source: ByteSource) -> bool:
    block = b""
    block_start = 0

    async def bytes_at(offset: int, length: int) -> bytes:
        nonlocal block, block_start
        if offset < block_start or offset + length > block_start + len(block):
            block_start = offset
            block = await source.read(offset, BLOCK_BYTES)
        return block[offset - block_start : offset - block_start + length]

    offset = len(PNG_SIGNATURE)
    while offset + 8 <= source.size:
        chunk_header = await bytes_at(offset, 8)
        chunk_length = read_uint32_be(chunk_header, 0)
        if starts_with_ascii(chunk_header, 4, "acTL"):
            return True
        if starts_with_ascii(chunk_header, 4, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > source.size:
            return False
        offset = next_offset
    return False


async def detect_supported_image_mime_type_of(source: ByteSource) -> Optional[str]:
    """``detect_supported_image_mime_type`` of a whole file, reading only what each check needs."""
    header = await source.read(0, HEADER_BYTES)
    if not starts_with(header, PNG_SIGNATURE):
        return detect_supported_image_mime_type(header)
    return "image/png" if is_png(header) and not (await _is_animated_png_of(source)) else None
