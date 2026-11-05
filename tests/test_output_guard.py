import io
import sys

from pi.coding_agent import output_guard


def teardown_function(_result):
    output_guard.restore_stdout()


def test_take_over_redirects_stdout_to_stderr(capsys):
    output_guard.take_over_stdout()
    assert output_guard.is_stdout_taken_over()
    print("stray")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "stray" in captured.err


def test_raw_stdout_keeps_original_stream(capsys):
    output_guard.take_over_stdout()
    output_guard.write_raw_stdout("protocol line\n")
    captured = capsys.readouterr()
    assert "protocol line" in captured.out
    assert captured.err == ""


def test_restore_returns_original_stdout():
    original = sys.stdout
    output_guard.take_over_stdout()
    output_guard.restore_stdout()
    assert sys.stdout is original
    assert not output_guard.is_stdout_taken_over()


def test_double_take_over_is_idempotent():
    original = sys.stdout
    output_guard.take_over_stdout()
    taken = sys.stdout
    output_guard.take_over_stdout()
    assert sys.stdout is taken
    output_guard.restore_stdout()
    assert sys.stdout is original
