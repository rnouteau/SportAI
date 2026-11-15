from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader

AnalysisT = TypeVar("AnalysisT", bound=Analysis[Any])


class Model(ABC, Generic[AnalysisT]):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def run(self, video_reader: VideoReader, analysis: AnalysisT) -> None:
        pass
