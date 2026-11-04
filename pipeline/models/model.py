
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, video_reader, analysis):
        pass