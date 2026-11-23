"""Unit tests for TennisSideStat (side of the court of every player, per frame)."""

from __future__ import annotations

import pytest

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_player_side_stat import TennisSideStat, side_of
from tests.helpers import make_tennis_analysis


def test_side_of() -> None:
    assert side_of(-12.0) == "near"
    assert side_of(15.0) == "far"
    assert side_of(0.0) is None
    assert side_of(None) is None


def test_side_is_written_on_every_frame() -> None:
    analysis = make_tennis_analysis(n_frames=3)
    for frame, ys in zip(analysis.frames, [(-12.0, 14.0), (-11.0, None), (3.0, 20.0)], strict=True):
        players = []
        for y in ys:
            player = TennisPlayerAtFrame(bbox=[0.0, 0.0, 10.0, 20.0])
            if y is not None:
                player.court = CourtPosition(x=0.0, y=y)
            players.append(player)
        frame.players = players

    TennisSideStat().run(None, analysis)  # type: ignore[arg-type]

    assert [[p.side for p in f.players] for f in analysis.frames] == [
        ["near", "far"],
        ["near", None],
        ["far", "far"],  # sides change with the players: no fixed assignment
    ]
    assert "side" not in analysis.frames[1].players[1].to_dict()


def test_side_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        TennisPlayerAtFrame(side="left")
    player = TennisPlayerAtFrame(side="far")
    assert player.to_dict()["side"] == "far"
    player.side = None
    assert "side" not in player.to_dict()
