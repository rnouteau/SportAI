"""Overlay drawing: colours and labels of the players, frames without a ball."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.render.tennis_overlay import (
    BALL_COLOR,
    ID_PALETTE,
    PLAYER_COLOR,
    _draw_ball,
    _draw_players,
    _player_label,
    player_color,
)


def _draw(*players: TennisPlayerAtFrame) -> np.ndarray:
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_players(canvas, SimpleNamespace(players=list(players)))
    return canvas


def test_player_color_is_stable_per_id() -> None:
    assert player_color(None) == PLAYER_COLOR
    assert player_color(3) == player_color(3) == ID_PALETTE[3]
    assert player_color(1) != player_color(2)
    assert player_color(len(ID_PALETTE) + 1) == ID_PALETTE[1]


def test_player_label() -> None:
    assert _player_label(1, 7) == "P1 #7"
    assert _player_label(2, None) == "P2"
    assert _player_label(None, 7) == "ID 7"
    assert _player_label(None, None) == ""


def test_player_drawn_with_the_colour_of_its_id_and_a_label() -> None:
    canvas = _draw(TennisPlayerAtFrame(bbox=[40.0, 60.0, 80.0, 140.0], track_id=7))
    color = np.array(player_color(7))
    assert (canvas[100, 40] == color).all()  # left edge of the box
    label_area = canvas[40:58, 40:90].reshape(-1, 3)
    assert (label_area == color).all(axis=1).sum() > 50  # filled label background above the box


def test_player_without_id_has_no_label() -> None:
    canvas = _draw(TennisPlayerAtFrame(bbox=[40.0, 60.0, 80.0, 140.0]))
    assert (canvas[100, 40] == np.array(PLAYER_COLOR)).all()
    assert not canvas[40:58, 40:90].any()


def test_selected_player_coloured_by_player_id() -> None:
    canvas = _draw(TennisPlayerAtFrame(bbox=[40.0, 60.0, 80.0, 140.0], track_id=7, player_id=1))
    assert (canvas[100, 40] == np.array(player_color(1))).all()


def test_label_stays_inside_the_image_for_a_box_at_the_top() -> None:
    canvas = _draw(TennisPlayerAtFrame(bbox=[180.0, 0.0, 199.0, 50.0], track_id=2))
    assert (canvas[0:20, :].reshape(-1, 3) == np.array(player_color(2))).all(axis=1).any()


def test_ball_is_drawn_and_a_frame_without_ball_is_left_untouched() -> None:
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    _draw_ball(canvas, SimpleNamespace(ball=None))
    assert not canvas.any()

    _draw_ball(canvas, SimpleNamespace(ball=TennisBall(position=[50.0, 50.0])))
    assert (canvas == np.array(BALL_COLOR)).all(axis=2).any()
