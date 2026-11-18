"""Integration: player detection match_prediction / compute via fake pipeline model."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.run import run_pipeline
from pipeline.video_reader import VideoReader
from tests.evaluation.player_detection_evaluation import PlayerDetectionEvaluator
from tests.helpers import make_synthetic_video


class FakeTennisPlayerModel(Model[TennisAnalysis]):
    """Writes one player bbox on even frames, none on odd frames."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            assert frame.id is not None
            if frame.id % 2 == 0:
                frame.players = [
                    TennisPlayerAtFrame.create(bbox=(10.0, 10.0, 50.0, 80.0), confidence=0.9)
                ]
            else:
                frame.players = []


@pytest.mark.integration
def test_player_evaluator_match_and_compute_on_synthetic_frames(tmp_path: Path) -> None:
    n_frames = 4
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": (
                    "tests.integration.test_player_detection_evaluator_flow.FakeTennisPlayerModel"
                ),
                "config": {},
            }
        ],
        "evaluation": {"iou_threshold": 0.5},
    }
    data = run_pipeline("tennis", str(video_path), config, test=True)
    assert data is not None

    # Frames 0,2: GT matches pred → TP; frame 1: GT only → FN; frame 3: empty → TN
    gt_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x0": [10.0, 10.0, 10.0],
            "y0": [10.0, 10.0, 10.0],
            "x1": [50.0, 50.0, 50.0],
            "y1": [80.0, 80.0, 80.0],
        }
    )

    evaluator = PlayerDetectionEvaluator(config, directory=tmp_path)
    frame_preds, frame_gts = evaluator._boxes_by_frame(data["frames"], gt_df)
    # Ensure empty frame 3 is present for TN (pipeline should emit all frames)
    assert 3 in frame_preds

    evaluator.match_prediction(frame_preds, frame_gts)
    metrics = evaluator.compute()

    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP"] == 0
    assert metrics["Total"] == 4
    assert metrics["Accuracy"] == pytest.approx(0.75)
    assert metrics["Precision"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(2 / 3)
    assert metrics["F1-Score"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
