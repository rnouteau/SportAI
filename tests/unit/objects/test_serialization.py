"""Analysis objects ↔ JSON: ``from_dict`` / ``to_dict``, files, empty frames, format rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipeline.objects.analysis import Analysis
from pipeline.objects.frame import Frame
from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_court_schema import TennisCourtSchema
from pipeline.objects.tennis.tennis_frame import TennisFrame
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player import TennisPlayer
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from tests.fakes import FakeVideoReader

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tennis_analysis_sample.json"
EMPTY_TENNIS_FRAME = {"players": [], "ball": None, "homography": None}


@pytest.fixture
def sample() -> dict:
    """A real pipeline output: 5 frames of Fonseca vs Djokovic + one empty frame."""
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


class _BareAnalysis(Analysis[Frame]):
    frame_class = Frame


# --------------------------------------------------------------------------- whole analysis
def test_sample_round_trip_is_lossless(sample: dict) -> None:
    analysis = TennisAnalysis.from_dict(copy.deepcopy(sample))
    assert analysis.to_dict() == sample


def test_save_then_load_gives_the_same_document(sample: dict, tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "analysis.json"
    TennisAnalysis.from_dict(copy.deepcopy(sample)).save_json(path)
    assert TennisAnalysis.load_json(path).to_dict() == sample


def test_loaded_objects_are_typed(sample: dict) -> None:
    analysis = TennisAnalysis.from_dict(sample)
    frame = analysis.frames[0]
    assert isinstance(frame, TennisFrame)
    assert isinstance(frame.ball, TennisBall)
    assert isinstance(frame.homography, TennisHomography)
    assert all(isinstance(p, TennisPlayerAtFrame) for p in frame.players)
    assert isinstance(frame.players[0].court, CourtPosition)
    assert all(isinstance(p, TennisPlayer) for p in analysis.final_stats.players)
    assert isinstance(analysis.court_schema, TennisCourtSchema)


def test_numpy_values_are_saved_as_plain_json(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    analysis = TennisAnalysis(sport="tennis", video={"fps": np.float32(30.0)})
    analysis.save_json(tmp_path / "a.json")
    assert json.loads((tmp_path / "a.json").read_text())["video"]["fps"] == 30.0


def test_old_file_without_final_stats_loads() -> None:
    analysis = TennisAnalysis.from_dict({"sport": "tennis", "video": {}, "frames": []})
    assert analysis.final_stats.players == []
    assert analysis.to_dict()["final_stats"] == {"players": []}


# --------------------------------------------------------------------------- empty frames
def test_empty_frames() -> None:
    assert Frame(id=2).to_dict() == {"id": 2}
    assert TennisFrame(id=1).to_dict() == {"id": 1, **EMPTY_TENNIS_FRAME}


def test_from_video_builds_one_empty_frame_per_video_frame() -> None:
    tennis = TennisAnalysis.from_video(FakeVideoReader(n_frames=3), "tennis")  # type: ignore[arg-type]
    assert [f.to_dict() for f in tennis.frames] == [
        {"id": i, **EMPTY_TENNIS_FRAME} for i in range(3)
    ]
    assert tennis.video["frame_count"] == 3

    bare = _BareAnalysis.from_video(FakeVideoReader(n_frames=2), "generic")  # type: ignore[arg-type]
    assert [f.to_dict() for f in bare.frames] == [{"id": 0}, {"id": 1}]


# --------------------------------------------------------------------------- frame content
def test_ball_and_homography_are_null_when_missing() -> None:
    frame = TennisFrame.from_dict({"id": 0, "ball": {"position": [1, 2], "confidence": 0.5}})
    assert frame.ball is not None
    frame.ball = None
    assert frame.to_dict() == {"id": 0, **EMPTY_TENNIS_FRAME}


def test_ball_fields() -> None:
    ball = TennisBall.from_dict({"position": [1, 2], "confidence": 0.8, "velocity": 3})
    assert (ball.position, ball.confidence, ball.velocity) == ([1, 2], 0.8, 3)
    assert ball.to_dict() == {"position": [1.0, 2.0], "confidence": 0.8, "velocity": 3.0}
    assert TennisBall(position=[3, 4]).to_dict() == {"position": [3.0, 4.0]}


def test_homography_writes_only_what_it_has() -> None:
    assert TennisHomography(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to_dict() == {
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
    assert TennisHomography().to_dict() == {}


# --------------------------------------------------------------------------- players
def test_player_none_fields_are_left_out() -> None:
    """The tracking evaluator reads ``player.get("player_id", player.get("track_id"))``:
    a ``"player_id": null`` would hide the track id."""
    player = TennisPlayerAtFrame(bbox=[1, 2, 3, 4], track_id=0)
    assert player.to_dict() == {"bbox": [1.0, 2.0, 3.0, 4.0], "track_id": 0}  # 0 is kept
    assert TennisPlayerAtFrame().to_dict() == {}


def test_player_full_round_trip() -> None:
    data = {
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "confidence": 0.9,
        "track_id": 7,
        "player_id": 1,
        "legs": {"court": {"x": 1.5, "y": -11.0}},
        "side": "near",
    }
    player = TennisPlayerAtFrame.from_dict(data)
    assert player.court == CourtPosition(x=1.5, y=-11.0)
    assert player.to_dict() == data


def test_player_box_under_its_old_name_position() -> None:
    player = TennisPlayerAtFrame.from_dict({"position": [1, 2, 3, 4]})
    assert player.bbox == [1, 2, 3, 4]
    assert "position" not in player.to_dict()


def test_player_feet_is_the_bottom_centre_of_the_box() -> None:
    feet = TennisPlayerAtFrame(bbox=[10.0, 20.0, 30.0, 40.0]).feet
    assert feet is not None and (feet.x, feet.y) == (20.0, 40.0)
    assert TennisPlayerAtFrame().feet is None


def test_player_side_is_validated() -> None:
    with pytest.raises(ValueError):
        TennisPlayerAtFrame(side="left")
    assert TennisPlayerAtFrame(side="far").to_dict() == {"side": "far"}


def test_final_stats_players() -> None:
    analysis = TennisAnalysis(sport="tennis", video={})
    analysis.final_stats.players = [TennisPlayer(id=1, name="player A", track_ids=[2, 55])]
    assert analysis.to_dict()["final_stats"] == {
        "players": [{"id": 1, "name": "player A", "track_ids": [2, 55]}]
    }


# --------------------------------------------------------------------------- positions
def test_court_position_distances() -> None:
    a, b = CourtPosition(x=3.0, y=4.0), CourtPosition(x=0.0, y=0.0)
    assert a.distance == pytest.approx(5.0)
    assert a.distance_to(b) == pytest.approx(5.0)
    assert CourtPosition.from_dict(a.to_dict()) == a
