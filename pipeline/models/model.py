from abc import ABC, abstractmethod

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader


class Model(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def run(self, video_reader: VideoReader, analysis: Analysis) -> None:
        pass
