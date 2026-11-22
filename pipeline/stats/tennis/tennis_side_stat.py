from __future__ import annotations

from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import FAR, NEAR
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader


def side_of(court_y: float | None) -> str | None:
    """``"near"`` for court y < 0 (camera side), ``"far"`` for y > 0, None on the net line."""
    if court_y is None or court_y == 0.0:
        return None
    return NEAR if court_y < 0.0 else FAR


class TennisSideStat(Stat[TennisAnalysis]):
    """Write ``player.side`` on every frame from the feet on the court (``legs.court.y``).

    Needs ``TennisLegsCourtStat`` before it. Players without a court position (no
    homography on that frame) get no side.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        logger.info("TennisSideStat initialized")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisSideStat")
        n_sides = 0
        for frame in analysis.frames:
            for player in frame.players:  # views on frame.data: the side is written in place
                court = player.legs.court
                player.side = side_of(court.y if court is not None else None)
                n_sides += player.side is not None
        if n_sides == 0:
            logger.warning("No player with a court position: run TennisLegsCourtStat first")
