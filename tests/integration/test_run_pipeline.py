"""run_pipeline end to end with fake models: JSON shape, stats, final_stats, overlay."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import pytest

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.render.tennis_overlay import render_overlay_video
from pipeline.run import run_pipeline
from pipeline.video_reader import VideoReader
from tests.fakes import FakeSceneModel
from tests.helpers import pipeline_config

BALL = ("tests.fakes.FakeBallModel", {"position": [10.0, 20.0]})
SCENE = ("tests.fakes.FakeSceneModel", {})
PLAYER_STATS = [
    "pipeline.stats.tennis.TennisLegsCourtStat",
    "pipeline.stats.tennis.TennisSideStat",
    "pipeline.stats.tennis.TennisPlayerSelectorStat",
]


def test_returns_the_analysis_document(synthetic_video: Callable[..., Path]) -> None:
    data = run_pipeline(
        "tennis", str(synthetic_video(n_frames=4)), pipeline_config([BALL]), test=True
    )

    assert data is not None
    assert data["sport"] == "tennis"
    assert set(data["video"]) >= {"video_path", "frame_count", "fps", "height", "width"}
    assert [f["id"] for f in data["frames"]] == [0, 1, 2, 3]
    assert data["frames"][0]["ball"] == {"position": [10.0, 20.0], "confidence": 0.9}
    assert data["final_stats"] == {"players": []}


def test_stats_run_after_the_models(synthetic_video: Callable[..., Path]) -> None:
    config = pipeline_config([BALL], stats=["tests.fakes.BallCountStat"])
    data = run_pipeline("tennis", str(synthetic_video(n_frames=3)), config, test=True)
    assert data is not None
    assert data["final_stats"]["players"][0]["name"] == "3 balls"


def test_player_stats_select_the_two_players(synthetic_video: Callable[..., Path]) -> None:
    """Tracker → feet on the court → side → selector, with a fake tracker and homography.

    The homography maps a 320x240 image onto the court: image y=240 → court y=-12 (near
    baseline), image y=0 → court y=+12 (far baseline). Two walking players and a still
    line judge on the far side; the line judge must be dropped.
    """
    n_frames, fps = 40, 10.0
    matrix = [[0.05, 0.0, -8.0], [0.0, -0.1, 12.0], [0.0, 0.0, 1.0]]
    tracks = []
    for f in range(n_frames):
        step = 2.0 * f
        near = [100.0 + step, 170.0, 120.0 + step, 230.0, 1]
        far = [220.0 - step, 15.0, 235.0 - step, 50.0, 2]
        judge = [300.0, 30.0, 310.0, 60.0, 3]
        tracks.append([near, far, judge])
    video = synthetic_video(n_frames=n_frames, size=(320, 240))
    config = pipeline_config(
        [("tests.fakes.FakeTrackingModel", {"tracks": tracks, "matrix": matrix})],
        stats=PLAYER_STATS,
    )
    assert cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) == pytest.approx(fps)

    data = run_pipeline("tennis", str(video), config, test=True)

    assert data is not None
    kept = {
        (p["track_id"], p["player_id"], p["side"]) for f in data["frames"] for p in f["players"]
    }
    assert kept == {(1, 1, "near"), (2, 2, "far")}
    assert data["final_stats"]["players"] == [
        {"id": 1, "name": "player A", "track_ids": [1]},
        {"id": 2, "name": "player B", "track_ids": [2]},
    ]


# --------------------------------------------------------------------------- overlay
def _overlay_config(tmp_path: Path, with_video: bool) -> tuple[dict, Path, Path]:
    json_path, overlay_path = tmp_path / "out" / "stats.json", tmp_path / "out" / "overlay.mp4"
    outputs: dict = {"path": str(json_path)}
    if with_video:
        outputs["video"] = {"path": str(overlay_path)}
    return pipeline_config([SCENE], outputs=outputs), json_path, overlay_path


def test_overlay_written_when_configured(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    config, json_path, overlay_path = _overlay_config(tmp_path, with_video=True)
    assert run_pipeline("tennis", str(synthetic_video(n_frames=3)), config, test=False) is None
    assert json_path.is_file()
    assert overlay_path.is_file() and overlay_path.stat().st_size > 0


def test_no_overlay_without_video_path(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    config, json_path, overlay_path = _overlay_config(tmp_path, with_video=False)
    run_pipeline("tennis", str(synthetic_video(n_frames=3)), config, test=False)
    assert json_path.is_file()
    assert not overlay_path.exists()


def test_no_overlay_in_test_mode(tmp_path: Path, synthetic_video: Callable[..., Path]) -> None:
    config, _, overlay_path = _overlay_config(tmp_path, with_video=True)
    assert run_pipeline("tennis", str(synthetic_video(n_frames=3)), config, test=True) is not None
    assert not overlay_path.exists()


def test_render_overlay_video_keeps_size_and_frame_count(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    reader = VideoReader(str(synthetic_video(n_frames=4)))
    analysis = TennisAnalysis.from_video(reader, "tennis")
    FakeSceneModel({}).run(reader, analysis)
    out_path = tmp_path / "overlay.mp4"
    render_overlay_video(reader, analysis, out_path)
    reader.release()

    cap = cv2.VideoCapture(str(out_path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert (written, size) == (4, (reader.width, reader.height))


# --------------------------------------------------------------------------- config
def test_config_must_name_the_analysis_class(synthetic_video: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="analysis"):
        run_pipeline("tennis", str(synthetic_video(n_frames=2)), {"models": []}, test=True)


def test_analysis_class_loaded_from_config(synthetic_video: Callable[..., Path]) -> None:
    data = run_pipeline("tennis", str(synthetic_video(n_frames=3)), pipeline_config([]), test=True)
    assert data is not None
    assert len(data["frames"]) == 3
