"""Shared metric helpers: Counts, box IoU, greedy box matching."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.evaluation.metrics import Counts, box_iou, greedy_match, match_counts, safe_div

COUNT = st.integers(min_value=0, max_value=1_000)


def test_safe_div() -> None:
    assert safe_div(1, 4) == 0.25
    assert safe_div(3, 0) == 0.0


def test_counts_scores() -> None:
    counts = Counts(tp=1, fp=1, fn=2, tn=1)
    assert counts.total == 5
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(1 / 3)
    assert counts.f1 == pytest.approx(0.4)
    assert counts.accuracy == pytest.approx(0.4)
    assert counts.scores() == {
        "Precision": counts.precision,
        "Recall": counts.recall,
        "F1-Score": counts.f1,
    }


def test_counts_add_and_empty() -> None:
    assert Counts(1, 2, 3, 4) + Counts(10, 20, 30, 40) == Counts(11, 22, 33, 44)
    empty = Counts()
    assert (empty.precision, empty.recall, empty.f1, empty.accuracy) == (0.0, 0.0, 0.0, 0.0)


@given(tp=COUNT, fp=COUNT, fn=COUNT, tn=COUNT)
def test_scores_stay_between_zero_and_one(tp: int, fp: int, fn: int, tn: int) -> None:
    counts = Counts(tp, fp, fn, tn)
    for value in (counts.precision, counts.recall, counts.f1, counts.accuracy):
        assert 0.0 <= value <= 1.0


def test_box_iou() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 20]) == pytest.approx(0.5)
    assert box_iou([10, 10, 0, 0], [0, 0, 10, 10]) == pytest.approx(1.0)  # corners in any order


def test_greedy_match_threshold_is_inclusive() -> None:
    assert greedy_match([[0, 0, 10, 10]], [[0, 0, 10, 20]], 0.5) == [(0, 0, pytest.approx(0.5))]
    assert greedy_match([[0, 0, 10, 10]], [[0, 0, 10, 21]], 0.5) == []


def test_greedy_match_is_one_to_one_and_takes_the_best_pair_first() -> None:
    gts = [[0, 0, 10, 10]]
    preds = [[0, 0, 10, 12], [0, 0, 10, 10]]  # the second prediction fits better
    assert greedy_match(preds, gts, 0.5) == [(1, 0, pytest.approx(1.0))]


def test_match_counts() -> None:
    assert match_counts(n_preds=3, n_gts=2, n_matches=2) == Counts(tp=2, fp=1, fn=0)
    assert greedy_match([], [[0, 0, 1, 1]], 0.5) == []
