"""Tennis non-regression tests: every suite of ``tests/nrt/suites.py``, one test each.

python -m pytest tests/nrt/tennis -m nrt -s                      # compare to the baselines
python -m pytest tests/nrt/tennis -m nrt -s -k players           # one suite
python -m pytest tests/nrt/tennis -m nrt -s --update-baseline    # rewrite the baselines
"""

from __future__ import annotations

import pytest

from tests.nrt.harness import run_suite
from tests.nrt.suites import SUITES, Suite


@pytest.mark.nrt
@pytest.mark.parametrize("suite", SUITES, ids=[suite.id for suite in SUITES])
def test_tennis_suite(suite: Suite, update_baseline: bool) -> None:
    outcome = run_suite(suite, update_baseline=update_baseline)
    if outcome.baseline_written:
        pytest.skip(f"{suite.id}: baseline written, review it then commit it")
    assert not outcome.problems, f"{suite.id} differs from the baseline:\n" + "\n".join(
        outcome.problems
    )
