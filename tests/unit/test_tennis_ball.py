import pytest

from pipeline.objects.ball import Ball
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_frame import TennisFrame


@pytest.mark.unit
def test_create_from_tuple(position=(10.5, 20), confidence=0.91) -> None:
    ball = TennisBall.create(position, confidence)

    assert isinstance(ball, TennisBall)
    assert isinstance(ball, Ball)
    assert ball.confidence == 0.91
    assert ball.position == [10.5, 20]
    assert ball.data == {"position": [10.5, 20], "confidence": 0.91}


@pytest.mark.unit
def test_create_from_list() -> None:
    position = [100, 200]
    confidence = 0.5
    ball = TennisBall.create(position, confidence)

    assert ball.position == [100, 200]


@pytest.mark.unit
def test_velocity_default_and_setter() -> None:
    ball = TennisBall.create(position=(1.0, 2.0), confidence=0.8)

    assert ball.velocity is None

    ball.velocity = 12.5

    assert ball.velocity == 12.5
    assert ball.data["velocity"] == 12.5

    ball.velocity = None

    assert ball.velocity is None
    assert ball.data["velocity"] is None


@pytest.mark.unit
def test_ball_is_a_view_over_the_dict() -> None:
    data = {"position": [1.0, 2.0], "confidence": 0.8}
    ball = TennisBall(data)

    data["position"] = [9.0, 8.0]
    data["confidence"] = 0.6

    assert ball.position == [9.0, 8.0]
    assert ball.confidence == 0.6


@pytest.mark.unit
def test_frame_ball_setter_writes_into_json() -> None:
    data = {"id": 0, "players": [], "ball": None}
    frame = TennisFrame(data)

    assert frame.ball is None

    frame.ball = TennisBall.create([10.0, 51.0], 0.76)

    assert frame.ball.position == [10.0, 51.0]
    assert frame.data["ball"] == {"position": [10.0, 51.0], "confidence": 0.76}

    frame.ball = None

    assert frame.ball is None
    assert frame.data["ball"] is None
