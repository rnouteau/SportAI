"""Unit tests for TennisSideStat (side of the court of every player, per frame)."""

from __future__ import annotations

import pytest

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_side_stat import TennisSideStat, side_of


@pytest.mark.unit
def test_side_of() -> None:
    assert side_of(-12.0) == "near"
    assert side_of(15.0) == "far"
    assert side_of(0.0) is None
    assert side_of(None) is None


@pytest.mark.unit
def test_side_is_written_on_every_frame() -> None:
    frames = [{"id": i, "players": [], "ball": None} for i in range(3)]
    analysis = TennisAnalysis({"sport": "tennis", "video": {}, "frames": frames})
    for frame, ys in zip(analysis.frames, [(-12.0, 14.0), (-11.0, None), (3.0, 20.0)], strict=True):
        players = []
        for y in ys:
            player = TennisPlayerAtFrame.create(position=[0.0, 0.0, 10.0, 20.0])
            if y is not None:
                player.legs.court = CourtPosition.create(x=0.0, y=y)
            players.append(player)
        frame.players = players

    TennisSideStat().run(None, analysis)  # type: ignore[arg-type]

    assert [[p.side for p in f.players] for f in analysis.frames] == [
        ["near", "far"],
        ["near", None],
        ["far", "far"],  # sides change with the players: no fixed assignment
    ]
    assert "side" not in analysis.frames[1].players[1].data


@pytest.mark.unit
def test_side_setter_rejects_unknown_values() -> None:
    player = TennisPlayerAtFrame.create()
    with pytest.raises(ValueError):
        player.side = "left"
    player.side = "far"
    player.side = None
    assert "side" not in player.data
