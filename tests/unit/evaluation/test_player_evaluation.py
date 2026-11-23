"""Player evaluator: MOT annotation reading, detection, identity and role metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.metrics import Counts
from tests.evaluation.player_evaluation import (
    GroundTruthBox,
    PlayerScores,
    PredictedBox,
    accumulate,
    distance_matrix,
    identity_metrics,
    load_mot_ground_truth,
    player_row,
    predictions_by_frame,
    score_detection_and_role,
)

NEAR = [100.0, 300.0, 150.0, 450.0]  # annotated id 1
FAR = [500.0, 100.0, 530.0, 180.0]  # annotated id 2


def _gt(n: int) -> dict[int, list[GroundTruthBox]]:
    return {f: [GroundTruthBox(NEAR, 1), GroundTruthBox(FAR, 2)] for f in range(n)}


def _preds(n: int, near: tuple, far: tuple, start: int = 0) -> dict[int, list[PredictedBox]]:
    """``near`` / ``far`` = ``(track_id, player_id)`` of each predicted box."""
    return {
        f: [PredictedBox(NEAR, *near), PredictedBox(FAR, *far)] for f in range(start, start + n)
    }


def _identity(gts: dict, preds: dict) -> dict:
    return identity_metrics([accumulate(gts, preds, 0.5)])


# --------------------------------------------------------------------------- reading
def test_load_mot_ground_truth_converts_boxes_and_drops_ignored_rows(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("1,2,10.00,20.00,30.00,100.00,1,1,1.0\n3,1,0,0,5,5,0,1,1.0\n")
    assert load_mot_ground_truth(gt) == {0: [GroundTruthBox([10.0, 20.0, 40.0, 120.0], 2)]}


def test_predictions_by_frame() -> None:
    frames = [
        {"id": 0, "players": [{"bbox": NEAR, "track_id": 7, "player_id": 1}, {"bbox": FAR}]},
        {"id": 1, "players": [{"confidence": 0.5}]},  # no box: ignored
    ]
    preds = predictions_by_frame(frames)
    assert preds == {0: [PredictedBox(NEAR, 7, 1), PredictedBox(FAR, None, None)], 1: []}
    assert preds[0][0].id == 1  # player_id wins over track_id
    assert PredictedBox(NEAR, 7, None).id == 7


def test_distance_matrix() -> None:
    preds = [[105.0, 305.0, 152.0, 452.0], [498.0, 102.0, 529.0, 181.0], [20.0, 200.0, 45.0, 240.0]]
    matrix = distance_matrix([NEAR, FAR], preds, 0.5)
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == pytest.approx(0.17, abs=0.01)
    assert matrix[1, 1] == pytest.approx(0.13, abs=0.01)
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[1, 0]) and np.isnan(matrix[:, 2]).all()
    assert distance_matrix([], [NEAR], 0.5).shape == (0, 1)


# --------------------------------------------------------------------------- detection + role
def test_detection_counts_and_mean_iou() -> None:
    preds = {0: [PredictedBox(NEAR, 1, 1), PredictedBox([0, 0, 5, 5], 9, None)], 1: []}
    scores = score_detection_and_role(_gt(2), preds, 0.5)
    assert scores.counts == Counts(tp=1, fp=1, fn=3)
    assert scores.mean_iou == pytest.approx(1.0)


def test_role_accuracy_sees_an_inversion_that_idf1_ignores() -> None:
    swapped = _preds(5, near=(7, 2), far=(3, 1))  # stable ids, but P1 is the far player
    scores = score_detection_and_role(_gt(5), swapped, 0.5)
    assert scores.role_accuracy == 0.0
    assert _identity(_gt(5), swapped)["IDF1"] == pytest.approx(1.0)

    right = score_detection_and_role(_gt(5), _preds(5, near=(7, 1), far=(3, 2)), 0.5)
    assert right.role_accuracy == 1.0


def test_role_is_only_checked_on_boxes_with_a_player_id() -> None:
    scores = score_detection_and_role(_gt(2), _preds(2, near=(7, None), far=(3, None)), 0.5)
    assert (scores.role_checked, scores.role_accuracy) == (0, 0.0)


# --------------------------------------------------------------------------- identity
def test_perfect_tracking_with_other_ids() -> None:
    metrics = _identity(_gt(5), _preds(5, near=(7, None), far=(3, None)))
    assert metrics == {"IDF1": pytest.approx(1.0), "ID switches": 0, "MOTA": pytest.approx(1.0)}
    assert isinstance(metrics["ID switches"], int)


def test_swapped_ids_count_switches() -> None:
    preds = {**_preds(3, (7, None), (3, None)), **_preds(3, (3, None), (7, None), start=3)}
    metrics = _identity(_gt(6), preds)
    assert metrics["ID switches"] == 2
    assert metrics["IDF1"] < 1.0


def test_boxes_without_id_are_left_out_of_the_identity_metrics() -> None:
    preds = _preds(2, near=(None, None), far=(3, None))
    metrics = _identity(_gt(2), preds)
    assert metrics["MOTA"] == pytest.approx(0.5)  # the near player counts as missed


# --------------------------------------------------------------------------- rows
def test_row_and_total() -> None:
    gts = _gt(2)
    preds = {**_preds(2, (7, 1), (3, 2)), 2: [PredictedBox(NEAR, 7, 1)]}  # close-up: 1 FP
    scores = score_detection_and_role(gts, preds, 0.5)
    scores.accumulators.append(accumulate(gts, preds, 0.5))
    row = player_row(scores)
    assert (row["GT"], row["TP"], row["FP"], row["FN"]) == (4, 4, 1, 0)
    assert row["Precision"] == pytest.approx(4 / 5)
    assert row["Role accuracy"] == 1.0
    assert not math.isnan(row["MOTA"])

    total = player_row(scores + scores)
    assert total["GT"] == 8
    assert total["IDF1"] == pytest.approx(row["IDF1"])


def test_empty_scores_add_up() -> None:
    assert (PlayerScores() + PlayerScores()).counts == Counts()
