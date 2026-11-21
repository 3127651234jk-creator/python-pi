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


def test_file_url_arguments_resolve(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello from url", encoding="utf-8")
    processed = asyncio.run(process_file_arguments([text.as_uri()], cwd=str(tmp_path),
                                                   auto_resize_images=False))
    assert "hello from url" in processed.text


def test_non_breaking_spaces_in_paths_are_normalized(tmp_path):
    write(tmp_path, "my notes.txt", b"hello nbsp")
    spaced = str(tmp_path / "my\u00a0notes.txt")
    processed = asyncio.run(process_file_arguments([spaced], cwd=str(tmp_path),
                                                   auto_resize_images=False))
    assert "hello nbsp" in processed.text



def test_invalid_utf8_text_is_decoded_with_replacement(tmp_path):
    # TS readFile(path, "utf-8") replaces invalid bytes; it does not fail the run.
    text = write(tmp_path, "latin1.txt", b"caf\xe9")
    processed = asyncio.run(process_file_arguments([text], cwd=str(tmp_path), auto_resize_images=False))
    assert "caf\ufffd" in processed.text
