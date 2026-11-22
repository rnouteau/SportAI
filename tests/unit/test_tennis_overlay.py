from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipeline.models.model import Model
from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.render.tennis_overlay import (
    ID_PALETTE,
    PLAYER_COLOR,
    _draw_players,
    _player_label,
    player_color,
    render_overlay_video,
)
from pipeline.run import run_pipeline
from pipeline.video_reader import VideoReader
from tests.helpers import make_synthetic_video


class FakeOverlayModel(Model[TennisAnalysis]):
    """Writes ball, player, and homography so the overlay has something to draw."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            ball = TennisBall.create()
            ball.position = (20.0, 15.0)
            ball.confidence = 0.95
            frame.ball = ball

            player = TennisPlayerAtFrame.create()
            player.bbox = (5.0, 5.0, 25.0, 40.0)
            player.confidence = 0.8
            player.track_id = 3
            player.legs.court = CourtPosition.create(x=1.0, y=2.0)
            frame.players = [player]

            homo = TennisHomography.create()
            homo.keypoints = [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]
            # Approximate image→court as identity scaled (invertible for line overlay).
            homo.matrix = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            frame.homography = homo


@pytest.mark.unit
def test_render_overlay_video_writes_mp4(tmp_path: Path) -> None:
    n_frames = 4
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    out_path = tmp_path / "overlay.mp4"

    reader = VideoReader(str(video_path))
    analysis = TennisAnalysis.from_video(reader, "tennis")
    FakeOverlayModel({}).run(reader, analysis)
    render_overlay_video(reader, analysis, out_path)
    reader.release()

    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert written == n_frames
    assert width == reader.width
    assert height == reader.height


@pytest.mark.unit
def test_run_pipeline_writes_overlay_when_configured(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3)
    overlay_path = tmp_path / "out" / "overlay.mp4"
    json_path = tmp_path / "out" / "stats.json"
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.unit.test_tennis_overlay.FakeOverlayModel",
                "config": {},
            }
        ],
        "outputs": {
            "path": str(json_path),
            "video": {"path": str(overlay_path)},
        },
    }

    result = run_pipeline("tennis", str(video_path), config, test=False)

    assert result is None
    assert overlay_path.is_file()
    assert overlay_path.stat().st_size > 0
    assert json_path.is_file()


@pytest.mark.unit
def test_run_pipeline_skips_overlay_when_video_path_absent(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3)
    json_path = tmp_path / "stats.json"
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.unit.test_tennis_overlay.FakeOverlayModel",
                "config": {},
            }
        ],
        "outputs": {"path": str(json_path)},
    }

    run_pipeline("tennis", str(video_path), config, test=False)

    assert json_path.is_file()
    assert list(tmp_path.glob("**/*.mp4")) == [video_path]


@pytest.mark.unit
def test_run_pipeline_skips_overlay_in_test_mode(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3)
    overlay_path = tmp_path / "overlay.mp4"
    config = {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [
            {
                "name": "tests.unit.test_tennis_overlay.FakeOverlayModel",
                "config": {},
            }
        ],
        "outputs": {
            "path": str(tmp_path / "stats.json"),
            "video": {"path": str(overlay_path)},
        },
    }

    data = run_pipeline("tennis", str(video_path), config, test=True)

    assert data is not None
    assert not overlay_path.exists()


@pytest.mark.unit
def test_player_color_is_stable_per_track_id() -> None:
    assert player_color(None) == PLAYER_COLOR
    assert player_color(3) == player_color(3) == ID_PALETTE[3]
    assert player_color(1) != player_color(2)
    assert player_color(len(ID_PALETTE) + 1) == ID_PALETTE[1]


def _frame_with_player(track_id: int | None) -> tuple[np.ndarray, object]:
    player = TennisPlayerAtFrame.create()
    player.bbox = (40.0, 60.0, 80.0, 140.0)
    player.track_id = track_id
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_players(canvas, SimpleNamespace(players=[player]))
    return canvas, player


@pytest.mark.unit
def test_players_drawn_with_id_color_and_label() -> None:
    canvas, _ = _frame_with_player(7)
    color = np.array(player_color(7))
    assert (canvas[100, 40] == color).all()  # left edge of the box
    label_area = canvas[40:58, 40:90].reshape(-1, 3)
    assert (label_area == color).all(axis=1).sum() > 50  # filled label background above the box


@pytest.mark.unit
def test_player_without_track_id_has_no_label() -> None:
    canvas, _ = _frame_with_player(None)
    assert (canvas[100, 40] == np.array(PLAYER_COLOR)).all()
    assert not canvas[40:58, 40:90].any()


@pytest.mark.unit
def test_label_stays_inside_image_for_box_at_top_edge() -> None:
    player = TennisPlayerAtFrame.create()
    player.bbox = (180.0, 0.0, 199.0, 50.0)
    player.track_id = 2
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_players(canvas, SimpleNamespace(players=[player]))
    assert (canvas[0:20, :].reshape(-1, 3) == np.array(player_color(2))).all(axis=1).any()


@pytest.mark.unit
def test_player_label() -> None:
    assert _player_label(1, 7) == "P1 #7"
    assert _player_label(2, None) == "P2"
    assert _player_label(None, 7) == "ID 7"
    assert _player_label(None, None) == ""


@pytest.mark.unit
def test_selected_player_colored_by_player_id() -> None:
    player = TennisPlayerAtFrame.create()
    player.bbox = (40.0, 60.0, 80.0, 140.0)
    player.track_id = 7
    player.player_id = 1
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_players(canvas, SimpleNamespace(players=[player]))
    assert (canvas[100, 40] == np.array(player_color(1))).all()
