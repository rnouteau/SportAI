
from abc import ABC, abstractmethod

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader


class Stat(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def setup(self, options):
        pass
    @abstractmethod
    def run(self,video_reader : VideoReader, analysis : Analysis):
        pass