"""Unit tests for the player tracking evaluator helpers (MOT metrics)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.player_tracking_evaluation import (
    accumulate,
    compute_metrics,
    distance_matrix,
    load_mot_ground_truth,
    predictions_by_frame,
)

P1 = [100.0, 300.0, 150.0, 450.0]
P2 = [500.0, 100.0, 530.0, 180.0]


@pytest.mark.unit
def test_distance_matrix_example() -> None:
    preds = [[105.0, 305.0, 152.0, 452.0], [498.0, 102.0, 529.0, 181.0], [20.0, 200.0, 45.0, 240.0]]
    matrix = distance_matrix([P1, P2], preds, 0.5)
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == pytest.approx(0.17, abs=0.01)
    assert matrix[1, 1] == pytest.approx(0.13, abs=0.01)
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[1, 0])
    assert np.isnan(matrix[:, 2]).all()


@pytest.mark.unit
def test_distance_matrix_empty_sides() -> None:
    assert distance_matrix([], [P1], 0.5).shape == (0, 1)
    assert distance_matrix([P1], [], 0.5).shape == (1, 0)


@pytest.mark.unit
def test_load_mot_ground_truth_columns_and_offset(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("1,2,10.00,20.00,30.00,100.00,1,1,1.0\n3,1,0,0,5,5,0,1,1.0\n")
    frames = load_mot_ground_truth(gt)
    assert frames == {0: [([10.0, 20.0, 40.0, 120.0], 2)]}  # ignored row dropped


@pytest.mark.unit
def test_predictions_skip_boxes_without_track_id() -> None:
    frames = [
        {"id": 0, "players": [{"position": P1, "track_id": 7}, {"position": P2}]},
        {"id": 1, "players": []},
    ]
    preds, without_id = predictions_by_frame(frames)
    assert preds == {0: [(P1, 7)]}
    assert without_id == 1


def _seq(n: int, ids: tuple[int, int]) -> dict[int, list[tuple[list[float], int]]]:
    return {f: [(P1, ids[0]), (P2, ids[1])] for f in range(n)}


@pytest.mark.unit
def test_perfect_tracking_with_other_ids() -> None:
    gts = _seq(5, (1, 2))
    preds = _seq(5, (7, 3))  # ids differ from the ground truth: only consistency matters
    m = compute_metrics(accumulate(gts, preds, 0.5))
    assert m["IDF1"] == pytest.approx(1.0)
    assert m["MOTA"] == pytest.approx(1.0)
    assert (m["ID switches"], m["FP"], m["FN"], m["GT"]) == (0, 0, 0, 10)
    assert isinstance(m["ID switches"], int) and isinstance(m["IDF1"], float)


@pytest.mark.unit
def test_swapped_ids_count_switches() -> None:
    gts = _seq(6, (1, 2))
    preds = {**_seq(3, (7, 3)), **{f: [(P1, 3), (P2, 7)] for f in range(3, 6)}}
    m = compute_metrics(accumulate(gts, preds, 0.5))
    assert m["ID switches"] == 2
    assert m["IDF1"] < 1.0


@pytest.mark.unit
def test_detection_on_frame_without_ground_truth_is_false_positive() -> None:
    gts = _seq(2, (1, 2))
    preds = {**_seq(2, (1, 2)), 2: [(P1, 1)]}  # close-up frame: nobody in the ground truth
    m = compute_metrics(accumulate(gts, preds, 0.5))
    assert m["FP"] == 1
    assert m["Precision"] == pytest.approx(4 / 5)
    assert not math.isnan(m["MOTA"])
