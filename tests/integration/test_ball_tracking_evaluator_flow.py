from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.run import run_pipeline
from pipeline.video_reader import VideoReader
from tests.evaluation.ball_detection_evaluation import BallTrackingEvaluator
from tests.helpers import make_synthetic_video


def make_evaluator(min_dist: int = 10) -> BallTrackingEvaluator:
    return BallTrackingEvaluator(
        config={"evaluation": {"min_dist": min_dist}},
        directory=Path("."),
    )


class FakeTennisBallModel(Model[TennisAnalysis]):
    """Writes a ball only on even frames, always at (100, 100)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            assert frame.id is not None
            if frame.id % 2 == 0:
                ball = TennisBall.create()
                ball.position = (100.0, 100.0)
                ball.confidence = 0.95
                frame.ball = ball
            else:
                frame.ball = None


@pytest.mark.integration
def test_evaluator_match_and_compute_on_synthetic_frames(tmp_path: Path) -> None:
    n_frames = 5
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": ("tests.integration.test_ball_tracking_evaluator_flow.FakeTennisBallModel"),
                "config": {},
            }
        ],
    }
    data = run_pipeline("tennis", str(video_path), config, test=True)

    assert data is not None

    gt_df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "x": [100.0, 100.0, 100.0, np.nan, np.nan],
            "y": [100.0, 100.0, 100.0, np.nan, np.nan],
        }
    )

    evaluator = make_evaluator()

    evaluator.match_prediction(data["frames"], gt_df)
    metrics = evaluator.compute()

    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP2"] == 1
    assert metrics["FP1"] == 0
    assert metrics["Total"] == 5
    assert metrics["FP"] == 1
    assert metrics["Accuracy"] == pytest.approx(0.6)
    assert metrics["Precision"] == pytest.approx(2 / 3)
    assert metrics["Recall"] == pytest.approx(2 / 3)
    assert metrics["F1-Score"] == pytest.approx(2 / 3)
