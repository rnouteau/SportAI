from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.evaluation.ball_tracking import BallTrackingEvaluator


def make_evaluator(min_dist: int = 10) -> BallTrackingEvaluator:
    return BallTrackingEvaluator(config={"evaluation": {"min_dist":min_dist}},
                                       directory = Path("."),
                                       )

@pytest.mark.unit
def test_match_prediction_counts_all_cases() -> None:
    """
    One frame per case:
      0 FN  : GT ball, no prediction
      1 TP  : GT ball, prediction within min_dist
      2 FP1 : GT ball, prediction too far
      3 FP2 : no GT ball, prediction present
      4 TN  : no GT ball, no prediction
    """
    frames = [
        {"id": 0, "ball": None},
        {"id": 1, "ball": {"position": [102, 100], "confidence": 0.9}},
        {"id": 2, "ball": {"position": [200, 200], "confidence": 0.9}},
        {"id": 3, "ball": {"position": [50, 50], "confidence": 0.9}},
        {"id": 4, "ball": None},
    ]
    gt_df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "x": [100.0, 100.0, 100.0, np.nan, np.nan],
            "y": [100.0, 100.0, 100.0, np.nan, np.nan],
        }
    )

    evaluator = make_evaluator(min_dist=10)
    evaluator.match_prediction(frames,gt_df)

    assert evaluator.tp == 1
    assert evaluator.fp1 == 1
    assert evaluator.fp2 == 1
    assert evaluator.tn == 1
    assert evaluator.fn == 1

@pytest.mark.unit
def test_compute_metrics_from_known_counts() -> None:
    evaluator = make_evaluator()
    evaluator.tp = 1
    evaluator.fp1 = 1
    evaluator.fp2 = 1
    evaluator.tn = 1
    evaluator.fn = 1


    metrics = evaluator.compute()

    assert metrics["Total"] == 5
    assert metrics["TP"] == 1
    assert metrics["FP1"] == 1
    assert metrics["FP2"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1
    assert metrics["Accuracy"] == pytest.approx(0.4)
    assert metrics["Precision"] == pytest.approx(1 / 3)
    assert metrics["Recall"] == pytest.approx(0.5)
    assert metrics["F1-Score"] == pytest.approx(0.4)

@pytest.mark.unit
def test_exact_hit_counts_as_tp() -> None:
    frames = [{"id": 0, "ball": {"position": [100, 100], "confidence": 1.0}}]
    gt_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0]})

    evaluator = make_evaluator(min_dist=10)
    evaluator.match_prediction(frames, gt_df)

    assert evaluator.tp == 1
    assert evaluator.fp1 == 0

@pytest.mark.unit
def test_distance_equal_to_min_dist_is_fp1() -> None:
    frames = [{"id": 0, "ball": {"position": [110, 100], "confidence": 1.0}}]
    gt_df = pd.DataFrame({"frame": [0], "x": [100.0], "y": [100.0]})

    evaluator = make_evaluator(min_dist=10)
    evaluator.match_prediction(frames, gt_df)

    assert evaluator.tp == 0
    assert evaluator.fp1 == 1

@pytest.mark.unit
def test_compute_with_zero_frames_does_not_crash() -> None:
    evaluator = make_evaluator()
    metrics = evaluator.compute()

    assert metrics["Total"] == 0
    assert metrics["Accuracy"] == pytest.approx(0.0)
    assert metrics["Precision"] == pytest.approx(0.0)
    assert metrics["Recall"] == pytest.approx(0.0)
    assert metrics["F1-Score"] == pytest.approx(0.0)