from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader


class TennisExampleStat(Stat[TennisAnalysis]):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        logger.info("TennisExampleStat initialized")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisExampleStat")
        detected = sum(1 for frame in analysis.frames if frame.ball)
        analysis.data.setdefault("stats", {})["ball_detections"] = detected
        for frame in analysis.frames:
            if frame.ball:
                logger.info(f"frame {frame.id}: ball at {frame.ball.position}")
