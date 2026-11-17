from pathlib import Path

import pytest

from tests.evaluation.player_detection_evaluation import (
    PlayerDetectionEvaluator,
    average_precision_at_iou,
    box_iou,
    greedy_match,
)


@pytest.mark.unit
def test_box_iou_perfect_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


@pytest.mark.unit
def test_box_iou_no_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


@pytest.mark.unit
def test_greedy_match_tp_when_iou_at_least_half() -> None:
    # IoU of these boxes is 0.5 exactly → TP at threshold 0.5
    pred = [[0, 0, 10, 10]]
    gt = [[0, 0, 10, 20]]
    assert box_iou(pred[0], gt[0]) == pytest.approx(0.5)
    tp, fp, fn = greedy_match(pred, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 0, 0)


@pytest.mark.unit
def test_greedy_match_fp_when_iou_below_half() -> None:
    pred = [[0, 0, 10, 10]]
    gt = [[0, 0, 10, 21]]  # IoU < 0.5
    assert box_iou(pred[0], gt[0]) < 0.5
    tp, fp, fn = greedy_match(pred, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (0, 1, 1)


@pytest.mark.unit
def test_match_prediction_empty_frame_is_tn() -> None:
    evaluator = PlayerDetectionEvaluator(
        {"evaluation": {"iou_threshold": 0.5}},
        directory=Path("."),
    )
    evaluator.match_prediction({0: []}, {0: []})
    assert evaluator.tn == 1
    assert evaluator.tp == evaluator.fp == evaluator.fn == 0


@pytest.mark.unit
def test_average_precision_bounds() -> None:
    preds = {0: [[0, 0, 10, 10]]}
    gts = {0: [[0, 0, 10, 10]]}
    ap = average_precision_at_iou(preds, gts, 0.5)
    assert 0.0 <= ap <= 1.0
    assert ap == pytest.approx(1.0)
