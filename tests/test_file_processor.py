import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.coding_agent.file_processor import process_file_arguments
from pi.ai.types import AudioContent, VideoContent


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_media_files_attach_as_content_blocks(tmp_path):
    wav = write(tmp_path, "clip.wav", b"RIFF....")
    mp4 = write(tmp_path, "clip.mp4", b"\x00\x00\x00\x18ftypmp42")
    processed = asyncio.run(process_file_arguments([wav, mp4], cwd=str(tmp_path), auto_resize_images=False))
    assert processed.media == [
        AudioContent(data=base64.b64encode(b"RIFF....").decode("ascii"), mime_type="audio/wav"),
        VideoContent(data=base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode("ascii"), mime_type="video/mp4"),
    ]
    assert processed.images == []
    assert f'"{wav}"' in processed.text and f'"{mp4}"' in processed.text


def test_unknown_extension_still_reads_as_text(tmp_path):
    text = write(tmp_path, "notes.txt", "hello".encode("utf-8"))
    processed = asyncio.run(process_file_arguments([text], cwd=str(tmp_path), auto_resize_images=False))
    assert processed.media == []
    assert "hello" in processed.text
