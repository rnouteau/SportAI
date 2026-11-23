"""BallEvaluator.run_test end to end: synthetic clip + CSV annotation, fake ball model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.evaluation.ball_evaluation import BallEvaluator
from tests.helpers import pipeline_config


def test_ball_evaluator_on_a_synthetic_clip(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    video = synthetic_video(n_frames=5)
    # The fake model writes a ball at (100, 100) on frames 0, 2, 4.
    video.with_suffix(".csv").write_text(
        "frame,x,y\n0,100,100\n1,100,100\n2,100,100\n3,,\n4,,\n", encoding="utf-8"
    )
    config = pipeline_config(
        [("tests.fakes.FakeBallModel", {"every": 2})], evaluation={"min_dist": 10}
    )

    rows = {row["Video"]: row for row in BallEvaluator(config, tmp_path).run_test()}

    clip = rows["clip.mp4"]
    # frames 0, 2: TP - frame 1: FN - frame 3: TN - frame 4: FP2
    assert (clip["TP"], clip["FN"], clip["TN"], clip["FP1"], clip["FP2"]) == (2, 1, 1, 0, 1)
    assert clip["Accuracy"] == pytest.approx(0.6)
    assert clip["Precision"] == pytest.approx(2 / 3)
    assert clip["Recall"] == pytest.approx(2 / 3)
    assert rows["TOTAL"]["TP"] == 2


def test_video_without_annotation_is_skipped(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    synthetic_video(n_frames=2)
    config = pipeline_config([("tests.fakes.FakeBallModel", {})])
    assert BallEvaluator(config, tmp_path).run_test() == []
