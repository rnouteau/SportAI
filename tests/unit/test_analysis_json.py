import pytest

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_frame import TennisFrame


def make_analysis() -> TennisAnalysis:
    return TennisAnalysis(
        {
            "sport": "tennis",
            "video": {
                "video_path": "fake.mp4",
                "fps": 30.0,
                "width": 1280,
                "height": 720,
                "frame_count": 2,
            },
            "frames": [
                {"id": 0, "players": [], "ball": None},
                {
                    "id": 1,
                    "players": [],
                    "ball": {"position": [10.0, 20.0], "confidence": 0.9},
                },
            ],
        }
    )


@pytest.mark.unit
def test_load_wraps_frames_as_tennis_frame() -> None:
    analysis = make_analysis()

    assert len(analysis.frames) == 2
    assert isinstance(analysis.frames[0], TennisFrame)
    assert analysis.frames[0].ball is None
    assert analysis.frames[1].ball is not None
    assert analysis.frames[1].ball.position == [10.0, 20.0]
    assert analysis.frames[1].ball.confidence == 0.9


@pytest.mark.unit
def test_save_and_load_round_trip(tmp_path) -> None:
    analysis = make_analysis()
    path = tmp_path / "subdir" / "analysis.json"
    analysis.save_json(path)

    assert path.exists

    loaded = TennisAnalysis.load_json(path)

    assert loaded.data["sport"] == "tennis"
    assert len(loaded.frames) == 2
    assert analysis.frames[0].ball is None
    assert analysis.frames[1].ball is not None
    assert analysis.frames[1].ball.position == [10.0, 20.0]
    assert analysis.frames[1].ball.confidence == 0.9


@pytest.mark.unit
def test_mutation_is_persisted_after_reload(tmp_path) -> None:
    analysis = make_analysis()
    analysis.frames[0].ball = TennisBall.create([0.5, 100], 0.7)
    path = tmp_path / "subdir" / "analysis.json"
    analysis.save_json(path)

    loaded = TennisAnalysis.load_json(path)
    assert loaded.frames[0].ball is not None
    assert loaded.frames[0].ball.position == [0.5, 100]
    assert loaded.frames[0].ball.confidence == 0.7
    assert loaded.data["frames"][0]["ball"] == {"position": [0.5, 100], "confidence": 0.7}


@pytest.mark.unit
def test_clearing_ball_writes_null_in_json(tmp_path) -> None:
    analysis = make_analysis()
    analysis.frames[1].ball = None
    path = tmp_path / "subdir" / "analysis.json"
    analysis.save_json(path)

    loaded = TennisAnalysis.load_json(path)
    assert loaded.frames[1].ball is None
    assert loaded.data["frames"][1]["ball"] is None
