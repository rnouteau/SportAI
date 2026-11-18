from __future__ import annotations

import numpy as np
from loguru import logger

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader


class TennisLegsCourtStat(Stat[TennisAnalysis]):
    """Project player legs onto the court via frame homography + analysis.court_schema."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        logger.info("TennisLegsCourtStat initialized")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisLegsCourtStat")
        schema = analysis.court_schema
        if schema is None:
            logger.error("No court_schema on analysis, cannot project legs")
            return

        for frame in analysis.frames:
            homo = frame.homography
            if homo is None or homo.matrix is None:
                continue
            for player in frame.players:
                image_pt = player.legs.image
                if image_pt is None:
                    continue
                player.legs.court = CourtPosition.from_image(
                    (image_pt.x, image_pt.y),
                    np.asarray(homo.matrix, dtype=np.float64),
                    schema,
                )
