"""Run only the abort test with a faulthandler stack dump on hang."""

import faulthandler
import sys

faulthandler.dump_traceback_later(15, exit=True)

import pytest

sys.exit(pytest.main(["tests/test_durable_harness.py::test_abort_generation_leaves_unanswered", "-q", "-p", "no:cacheprovider", "--tb=short"]))
