from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar
import json

from pipeline.objects.frame import Frame
from pipeline.video_reader import VideoReader

FrameT = TypeVar("FrameT", bound=Frame)


class Analysis(ABC, Generic[FrameT]):
    frame_class: type[FrameT]

    def __init__(self, data: dict) -> None:
        self.data = data
        self.frames: list[FrameT] = [self.frame_class(frame_data) for frame_data in data["frames"]]

    @classmethod
    def from_video(cls, video_reader: VideoReader, sport: str) -> "Analysis[FrameT]":
        frames = [{"id": i, "players": [], "ball": None} for i in range(video_reader.frame_count)]
        return cls({"sport": sport, "video": video_reader.info(), "frames": frames})

    @classmethod
    def load_json(cls, path: str | Path) -> "Analysis[FrameT]":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
