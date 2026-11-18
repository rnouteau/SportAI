import json
from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

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
        frames: list[dict] = [cls.frame_class.empty(i) for i in range(video_reader.frame_count)]
        return cls({"sport": sport, "video": video_reader.info(), "frames": frames})

    @classmethod
    def load_json(cls, path: str | Path) -> "Analysis[FrameT]":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, default=_json_default)


def _json_default(obj: object) -> object:
    """Convert NumPy scalars / arrays so json.dump does not raise TypeError."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
