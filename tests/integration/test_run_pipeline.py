"""run_pipeline end to end with fake models: JSON shape, stats, final_stats, overlay."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg
import pytest

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.render.tennis_overlay import render_overlay_video
from pipeline.run import analysis_id, run_pipeline
from pipeline.video_reader import VideoReader
from tests.fakes import FakeSceneModel
from tests.helpers import pipeline_config

BALL = ("tests.fakes.FakeBallModel", {"position": [10.0, 20.0]})
SCENE = ("tests.fakes.FakeSceneModel", {})
PLAYER_STATS = [
    "pipeline.stats.tennis.TennisLegsCourtStat",
    "pipeline.stats.tennis.TennisPlayerSideStat",
    "pipeline.stats.tennis.TennisPlayerSelectorStat",
    "pipeline.stats.tennis.TennisPlayerMovementStat",
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


N_FRAMES, FPS = 40, 10.0


def _two_players_config(**extra: object) -> dict:
    """Fake tracker + homography for a 320x240 clip of ``N_FRAMES`` frames, player stats on.

    The homography maps the image onto the court: image y=240 → court y=-12 (near
    baseline), image y=0 → court y=+12 (far baseline). Two players walk 2 px/frame
    (0.1 m/frame); a line judge stands still on the far side and must be dropped.
    """
    matrix = [[0.05, 0.0, -8.0], [0.0, -0.1, 12.0], [0.0, 0.0, 1.0]]
    tracks = []
    for f in range(N_FRAMES):
        step = 2.0 * f
        near = [100.0 + step, 170.0, 120.0 + step, 230.0, 1]
        far = [220.0 - step, 15.0, 235.0 - step, 50.0, 2]
        judge = [300.0, 30.0, 310.0, 60.0, 3]
        tracks.append([near, far, judge])
    return pipeline_config(
        [("tests.fakes.FakeTrackingModel", {"tracks": tracks, "matrix": matrix})],
        stats=PLAYER_STATS,
        **extra,
    )


def test_player_stats_select_the_two_players(synthetic_video: Callable[..., Path]) -> None:
    """Tracker → feet on the court → side → selector → movement."""
    video = synthetic_video(n_frames=N_FRAMES, size=(320, 240))
    config = _two_players_config()
    assert cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) == pytest.approx(FPS)

    data = run_pipeline("tennis", str(video), config, test=True)

    assert data is not None
    kept = {
        (p["track_id"], p["player_id"], p["side"]) for f in data["frames"] for p in f["players"]
    }
    assert kept == {(1, 1, "near"), (2, 2, "far")}
    players = data["final_stats"]["players"]
    assert [(p["id"], p["name"], p["track_ids"]) for p in players] == [
        (1, "player A", [1]),
        (2, "player B", [2]),
    ]
    # Both walk 2 px/frame = 0.1 m/frame at 10 fps: 1 m/s = 3.6 km/h for 3.9 s.
    for player in players:
        assert player["max_speed_kmh"] == pytest.approx(3.6, abs=0.1)
        assert player["distance_m"] == pytest.approx(3.9, abs=0.2)


# --------------------------------------------------------------------------- output folder
def test_analysis_id_is_a_clean_folder_and_url_name() -> None:
    name = "Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros 2026 [n2zjuuvgkGY]_cut_3354_3847.mp4"
    assert (
        analysis_id(name)
        == "fonseca-vs-djokovic-full-match-roland-garros-2026-n2zjuuvgkgy-_cut_3354_3847"
    )
    assert analysis_id("/videos/Clip.mp4") == "clip"
    assert analysis_id("｜｜.mp4") == "analysis"


def test_everything_goes_in_the_analysis_folder(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    video = synthetic_video(n_frames=N_FRAMES, size=(320, 240))  # clip.mp4 → id "clip"
    outputs = {"dir": str(tmp_path / "out"), "video": True, "heatmaps": True}

    assert run_pipeline("tennis", str(video), _two_players_config(outputs=outputs)) is None

    folder = tmp_path / "out" / "clip"
    assert sorted(p.name for p in folder.iterdir()) == ["heatmaps", "overlay.mp4", "stats.json"]
    assert sorted(p.name for p in (folder / "heatmaps").iterdir()) == [
        "heatmap_player_1.png",
        "heatmap_player_2.png",
    ]
    assert cv2.imread(str(folder / "heatmaps" / "heatmap_player_1.png")) is not None
    assert TennisAnalysis.load_json(folder / "stats.json").final_stats.players


def test_output_dir_argument_wins_over_the_config(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    config = pipeline_config([SCENE], outputs={"dir": str(tmp_path / "from_config")})
    run_pipeline(
        "tennis", str(synthetic_video(n_frames=3)), config, output_dir=str(tmp_path / "cli")
    )
    assert (tmp_path / "cli" / "clip" / "stats.json").is_file()
    assert not (tmp_path / "from_config").exists()


def test_visual_outputs_are_optional(tmp_path: Path, synthetic_video: Callable[..., Path]) -> None:
    config = pipeline_config([SCENE], outputs={"dir": str(tmp_path)})
    run_pipeline("tennis", str(synthetic_video(n_frames=3)), config)
    assert sorted(p.name for p in (tmp_path / "clip").iterdir()) == ["stats.json"]


def test_nothing_is_written_in_test_mode(
    tmp_path: Path, synthetic_video: Callable[..., Path]
) -> None:
    outputs = {"dir": str(tmp_path / "out"), "video": True, "heatmaps": True}
    video = synthetic_video(n_frames=N_FRAMES, size=(320, 240))
    assert run_pipeline("tennis", str(video), _two_players_config(outputs=outputs), test=True)
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- overlay
def test_render_overlay_video_is_h264_with_same_size_and_frames(
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

    # H.264 / yuv420p: what browsers play in a <video> tag (OpenCV's mp4v is not).
    meta = next(imageio_ffmpeg.read_frames(str(out_path)))
    assert meta["codec"] == "h264"
    assert meta["pix_fmt"].startswith("yuv420p")


# --------------------------------------------------------------------------- config
def test_config_must_name_the_analysis_class(synthetic_video: Callable[..., Path]) -> None:
    with pytest.raises(ValueError, match="analysis"):
        run_pipeline("tennis", str(synthetic_video(n_frames=2)), {"models": []}, test=True)


def test_analysis_class_loaded_from_config(synthetic_video: Callable[..., Path]) -> None:
    data = run_pipeline("tennis", str(synthetic_video(n_frames=3)), pipeline_config([]), test=True)
    assert data is not None
    assert len(data["frames"]) == 3
