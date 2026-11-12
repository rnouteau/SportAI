from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader


class ExampleStat(Stat):
    def __init__(self) -> None:
        super().__init__()
        logger.info("ExampleStat initialized")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running ExampleStat")
        for frame in analysis.frames:
            if frame.ball:
                logger.info(f"frame {frame.id}: ball at {frame.ball.position}")
