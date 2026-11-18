from pathlib import Path

import pytest

from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.run import run_pipeline
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader
from tests.helpers import make_synthetic_video


class FakeTennisBallModel(Model[TennisAnalysis]):
    """No weights / no TorchScript: writes a fake ball on every frame"""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            frame.ball = TennisBall.create(position=(10.0, 20.0), confidence=0.9)


class FakeBallCountStat(Stat[TennisAnalysis]):
    """Writes ball detection count into analysis.data['stats']."""

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        detected = sum(1 for frame in analysis.frames if frame.ball)
        analysis.data.setdefault("stats", {})["ball_detections"] = detected


@pytest.mark.integration
def test_run_pipeline_with_fake_model_writes_ball(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=5)
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.integration.test_run_pipeline_fake_model.FakeTennisBallModel",
                "config": {},
            }
        ],
    }

    data = run_pipeline("tennis", str(video_path), config, test=True)

    assert data is not None

    balls = [f["ball"] for f in data["frames"]]

    assert all(ball is not None for ball in balls)
    assert balls[0]["position"] == [10.0, 20.0]
    assert balls[0]["confidence"] == pytest.approx(0.9)


@pytest.mark.integration
def test_run_pipeline_returns_analysis_dict_shape(tmp_path: Path) -> None:
    n_frames = 4
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.integration.test_run_pipeline_fake_model.FakeTennisBallModel",
                "config": {},
            }
        ],
    }

    data = run_pipeline("tennis", str(video_path), config, test=True)

    assert data is not None
    assert data["sport"] == "tennis"
    assert "video" in data
    assert "frames" in data
    assert len(data["frames"]) == n_frames
    assert data["frames"][0]["id"] == 0
    assert set(data["video"]) >= {"video_path", "frame_count", "fps", "height", "width"}


@pytest.mark.integration
def test_run_pipeline_runs_stats_before_release(tmp_path: Path) -> None:
    n_frames = 3
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.integration.test_run_pipeline_fake_model.FakeTennisBallModel",
                "config": {},
            }
        ],
        "stats": [
            {
                "name": "tests.integration.test_run_pipeline_fake_model.FakeBallCountStat",
                "config": {},
            }
        ],
    }

    data = run_pipeline("tennis", str(video_path), config, test=True)

    assert data is not None
    assert data["stats"]["ball_detections"] == n_frames
