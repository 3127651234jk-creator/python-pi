import json

from pi.coding_agent import crash_log


class Boom(Exception):
    pass


def test_record_and_read(tmp_path):
    path = tmp_path / "crashes.json"
    record = crash_log.record_crash("uncaught_exception", Boom("exploded"), cwd=str(tmp_path), path=path)
    assert record["message"] == "exploded"
    assert record["kind"] == "uncaught_exception"
    assert "Boom" in record["stack"]
    assert record["sessionFile"] is None
    assert len(crash_log.read_crash_log(path)) == 1


def test_keeps_only_recent_records(tmp_path):
    path = tmp_path / "crashes.json"
    for index in range(7):
        crash_log.record_crash("uncaught_exception", Boom(f"crash-{index}"), cwd=".", path=path)
    records = crash_log.read_crash_log(path)
    assert len(records) == 5
    assert records[-1]["message"] == "crash-6"


def test_take_unnotified_marks_all_notified(tmp_path):
    path = tmp_path / "crashes.json"
    crash_log.record_crash("uncaught_exception", Boom("first"), cwd=".", path=path)
    crash = crash_log.take_unnotified_crash(path)
    assert crash is not None and crash["message"] == "first"
    assert crash_log.take_unnotified_crash(path) is None
    assert all(record.get("notified") for record in crash_log.read_crash_log(path))


def test_corrupt_file_is_tolerated(tmp_path):
    path = tmp_path / "crashes.json"
    path.write_text("{not json", encoding="utf-8")
    assert crash_log.read_crash_log(path) == []
    assert crash_log.record_crash("fatal_error", Boom("after"), cwd=".", path=path) is not None
    assert len(crash_log.read_crash_log(path)) == 1


def test_clear_crash_log(tmp_path):
    path = tmp_path / "crashes.json"
    crash_log.record_crash("uncaught_exception", Boom("x"), cwd=".", path=path)
    crash_log.clear_crash_log(path)
    assert crash_log.read_crash_log(path) == []


def test_extension_stack_match(tmp_path):
    stack = 'Traceback:\n  File "C:/proj/.pi/extensions/demo.py", line 1, in setup'
    assert crash_log.find_extension_stack_matches(stack, [str(tmp_path / "demo.py")]) == []
    assert crash_log.find_extension_stack_matches(stack, ["C:/proj/.pi/extensions/demo.py"]) == [
        "C:/proj/.pi/extensions/demo.py"]
    assert crash_log.find_extension_stack_matches(None, ["anything"]) == []


def test_invalid_kind_raises(tmp_path):
    try:
        crash_log.record_crash("other", Boom("x"), cwd=".", path=tmp_path / "c.json")
    except ValueError as exc:
        assert "Crash kind" in str(exc)
    else:
        raise AssertionError("expected ValueError")
