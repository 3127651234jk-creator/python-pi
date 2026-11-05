import sys

from pi.coding_agent import timings


def test_disabled_by_default(monkeypatch, capsys):
    monkeypatch.setattr(timings, "ENABLED", False)
    timings.reset_timings()
    timings.time("step")
    timings.print_timings()
    assert capsys.readouterr().err == ""


def test_records_and_prints(monkeypatch, capsys):
    monkeypatch.setattr(timings, "ENABLED", True)
    timings.reset_timings()
    timings.time("step-a", "main")
    timings.time("step-b", "main")
    timings.time("ext", "extensions")
    timings.print_timings()
    err = capsys.readouterr().err
    assert "--- Startup Timings: main ---" in err
    assert "step-a: 0ms" in err
    assert "step-b: 0ms" in err
    assert "TOTAL: 0ms" in err
    assert "--- Startup Timings: extensions ---" in err
    assert "ext: 0ms" in err


def test_reset_clears_namespace(monkeypatch, capsys):
    monkeypatch.setattr(timings, "ENABLED", True)
    timings.reset_timings()
    timings.time("first")
    timings.reset_timings()
    timings.time("second")
    timings.print_timings()
    err = capsys.readouterr().err
    assert "first" not in err
    assert "second: 0ms" in err
