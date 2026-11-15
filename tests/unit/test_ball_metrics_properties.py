from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from loguru import logger

from tests.evaluation.ball_tracking import BallTrackingEvaluator

COUNTS = st.integers(min_value=0, max_value=1_000)

logger.disable("tests.evaluation.ball_tracking")

_EVALUATOR = BallTrackingEvaluator(
    config={"evaluation": {"min_dist": 10}},
    directory=Path("."),
)


@pytest.mark.unit
@given(tp=COUNTS, fp1=COUNTS, fp2=COUNTS, tn=COUNTS, fn=COUNTS)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_compute_counts_sum_and_metrics_in_unit_interval(
    tp: int, fp1: int, fp2: int, tn: int, fn: int
) -> None:
    evaluator = _EVALUATOR
    evaluator.tp = tp
    evaluator.fp1 = fp1
    evaluator.fp2 = fp2
    evaluator.tn = tn
    evaluator.fn = fn

    metrics = evaluator.compute()

    assert metrics["Total"] == tp + fp1 + fp2 + tn + fn
    assert metrics["TP"] == tp
    assert metrics["FP1"] == fp1
    assert metrics["FP2"] == fp2
    assert metrics["FP"] == fp1 + fp2
    assert metrics["TN"] == tn
    assert metrics["FN"] == fn

    for name in ("Accuracy", "Precision", "Recall", "F1-Score"):
        value = metrics[name]
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0, 1]"
