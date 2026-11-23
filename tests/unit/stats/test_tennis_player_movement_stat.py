"""TennisPlayerMovementStat: distance and speed of the players, robust to box jitter."""

from __future__ import annotations

import random

import pytest

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_player import TennisPlayer
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_player_movement_stat import (
    MovementConfig,
    Point,
    TennisPlayerMovementStat,
    path_length,
    player_movement,
    smooth,
    speeds_kmh,
    split_at_gaps,
)
from tests.helpers import make_tennis_analysis

FPS = 50.0
CONFIG = MovementConfig()


def _still(n: int, jitter: float, seed: int = 0) -> list[Point]:
    """A player standing at (1, -11) whose position jitters by up to ``jitter`` metres."""
    rng = random.Random(seed)
    return [
        (f, 1.0 + rng.uniform(-jitter, jitter), -11.0 + rng.uniform(-jitter, jitter))
        for f in range(n)
    ]


def _walk(n: int, speed_ms: float, start: int = 0) -> list[Point]:
    """Straight line along x at ``speed_ms`` metres per second."""
    return [(f, (f - start) * speed_ms / FPS, -11.0) for f in range(start, start + n)]


def test_still_player_with_jitter_does_not_move() -> None:
    points = _still(int(5 * FPS), jitter=0.05)
    assert path_length(points) > 5.0  # raw steps: metres of pure noise
    movement = player_movement(points, FPS, CONFIG)
    assert movement.distance_m < 1.0
    assert movement.max_speed_kmh is not None and movement.max_speed_kmh < 2.0


def test_walk_at_3_metres_per_second() -> None:
    movement = player_movement(_walk(int(4 * FPS), speed_ms=3.0), FPS, CONFIG)
    assert movement.duration_s == pytest.approx(4.0, abs=0.05)
    assert movement.distance_m == pytest.approx(12.0, abs=0.3)  # ends lose half a window
    assert movement.max_speed_kmh == pytest.approx(10.8, abs=0.1)
    assert movement.mean_speed_kmh == pytest.approx(10.8, abs=0.5)


def test_a_gap_is_not_counted_as_a_run() -> None:
    before = _walk(50, speed_ms=0.0)  # standing at x = 0 ...
    after = [(f + 200, 20.0, -11.0) for f in range(50)]  # ... back 3 s later, 20 m away
    movement = player_movement(before + after, FPS, CONFIG)
    assert movement.distance_m == pytest.approx(0.0)
    assert movement.max_speed_kmh == pytest.approx(0.0)


def test_split_at_gaps() -> None:
    points = [(0, 0.0, 0.0), (1, 0.0, 0.0), (5, 0.0, 0.0), (40, 0.0, 0.0)]
    assert [[p[0] for p in s] for s in split_at_gaps(points, max_gap_frames=25)] == [
        [0, 1, 5],
        [40],
    ]


def test_smooth_averages_over_the_window() -> None:
    points = [(0, 0.0, 0.0), (1, 3.0, 0.0), (2, 0.0, 0.0)]
    assert smooth(points, half_window_frames=1)[1] == (1, 1.0, 0.0)
    assert smooth(points, half_window_frames=0) == points


def test_speed_uses_the_real_time_between_positions() -> None:
    points = [(0, 0.0, 0.0), (10, 1.0, 0.0)]  # 1 m in 10 frames = 0.2 s
    assert speeds_kmh(points, window_frames=5, fps=FPS) == [pytest.approx(18.0)]
    assert speeds_kmh(points[:1], window_frames=5, fps=FPS) == []


def test_config_from_dict() -> None:
    assert MovementConfig.from_dict({"smoothing_seconds": 1}).smoothing_seconds == 1.0
    assert MovementConfig.from_dict(None) == MovementConfig()


def test_stat_writes_the_final_players() -> None:
    n = int(2 * FPS)
    analysis = make_tennis_analysis(n_frames=n, fps=FPS)
    for frame in analysis.frames:
        x = frame.id * 2.0 / FPS  # player 1 walks at 2 m/s
        frame.players = [
            TennisPlayerAtFrame(player_id=1, court=CourtPosition(x=x, y=-11.0)),
            TennisPlayerAtFrame(player_id=2, court=None),  # never on the court
            TennisPlayerAtFrame(player_id=None, court=CourtPosition(x=0.0, y=0.0)),
        ]
    analysis.final_stats.players = [TennisPlayer(id=1), TennisPlayer(id=2)]

    TennisPlayerMovementStat({}).run(None, analysis)  # type: ignore[arg-type]

    walker, absent = analysis.final_stats.players
    assert walker.max_speed_kmh == pytest.approx(7.2, abs=0.1)
    assert walker.distance_m == pytest.approx(4.0, abs=0.3)
    assert absent.to_dict() == {"id": 2, "track_ids": []}  # nothing written, no null


def test_without_final_players_nothing_happens() -> None:
    analysis = make_tennis_analysis(n_frames=2)
    TennisPlayerMovementStat().run(None, analysis)  # type: ignore[arg-type]
    assert analysis.final_stats.players == []
