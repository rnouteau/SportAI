

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader
from loguru import logger

class ExampleStat(Stat):
    def __init__(self):
        super().__init__()
        logger.info("ExampleStat initialized")

    def run(self, video_reader: VideoReader, analysis : TennisAnalysis):
        logger.info("Running ExampleStat")
        # Implement the logic for the stat here
        for frame in analysis.frames:
            pass
    