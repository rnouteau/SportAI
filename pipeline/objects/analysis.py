

from abc import ABC
import os

from pipeline.objects.frame import Frame
from typing import Generic,TypeVar
import json

from pipeline.video_reader import VideoReader

FrameT= TypeVar(name="FrameT",bound=Frame)
class Analysis(ABC,Generic[FrameT]):
    frame_class : type[FrameT]

    def __init__(self, data: dict):
        self.data = data
        self.frames : list[FrameT] = [self.frame_class(frame_data) for frame_data in data["frames"]]

    @classmethod
    def from_video(cls, video_reader: VideoReader, sport: str) -> "Analysis":
        frames = [{"id": i, "players": [], "ball": None} for i in range(video_reader.frame_count)]

        return cls({"sport": sport,"video":video_reader.info(), "frames": frames})

    @classmethod
    def load_json(cls, path: str) -> "Analysis":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    def save_json(self, dir: str) -> None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir,"stats.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4)
