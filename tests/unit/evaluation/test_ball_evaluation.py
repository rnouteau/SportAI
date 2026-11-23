"""Ball evaluator: per-frame matching (TP / FP1 / FP2 / FN / TN) and the result row."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.evaluation.ball_evaluation import BallCounts, match_balls, predicted_balls


def _gt(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def _frame(frame_id: int, position: list[float] | None) -> dict:
    ball = None if position is None else {"position": position, "confidence": 0.9}
    return {"id": frame_id, "ball": ball}


def test_every_case_once() -> None:
    frames = [
        _frame(0, None),  # annotated, nothing predicted     -> FN
        _frame(1, [102, 100]),  # annotated, close             -> TP
        _frame(2, [200, 200]),  # annotated, too far           -> FP1
        _frame(3, [50, 50]),  # not annotated, predicted       -> FP2
        _frame(4, None),  # not annotated, nothing predicted   -> TN
    ]
    gt = _gt(
        [(0, 100, 100), (1, 100, 100), (2, 100, 100), (3, np.nan, np.nan), (4, np.nan, np.nan)]
    )
    assert match_balls(frames, gt, min_dist=10) == BallCounts(tp=1, fp1=1, fp2=1, fn=1, tn=1)


def test_distance_equal_to_min_dist_is_too_far() -> None:
    gt = _gt([(0, 100.0, 100.0)])
    assert match_balls([_frame(0, [100, 100])], gt, min_dist=10).tp == 1
    assert match_balls([_frame(0, [110, 100])], gt, min_dist=10).fp1 == 1


def test_frames_without_annotation_are_ignored() -> None:
    gt = _gt([(0, 100.0, 100.0)])
    counts = match_balls([_frame(0, [100, 100]), _frame(1, [5, 5])], gt, min_dist=10)
    assert counts == BallCounts(tp=1)


def test_predicted_balls() -> None:
    assert predicted_balls([_frame(0, None), _frame(1, [3, 4])]) == {1: (3.0, 4.0)}


def test_row_from_known_counts() -> None:
    row = BallCounts(tp=1, fp1=1, fp2=1, fn=1, tn=1).row()
    assert (row["Total"], row["TP"], row["FP"], row["TN"], row["FN"]) == (5, 1, 2, 1, 1)
    assert row["Accuracy"] == pytest.approx(0.4)
    assert row["Precision"] == pytest.approx(1 / 3)
    assert row["Recall"] == pytest.approx(0.5)
    assert row["F1-Score"] == pytest.approx(0.4)


def test_counts_add_and_empty_row() -> None:
    total = BallCounts(tp=1, fn=2) + BallCounts(tp=3, fp2=1)
    assert total == BallCounts(tp=4, fp2=1, fn=2)
    assert BallCounts().row()["F1-Score"] == 0.0
