import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar, cast

from pipeline.objects.court_schema import CourtSchema
from pipeline.objects.final_stats import FinalStats
from pipeline.objects.frame import Frame
from pipeline.video_reader import VideoReader

FrameT = TypeVar("FrameT", bound=Frame)


@dataclass
class Analysis(Generic[FrameT]):
    frame_class: ClassVar[type[Frame]]
    final_stats_class: ClassVar[type[FinalStats]] = FinalStats
    sport: str
    video: dict
    frames: list[FrameT] = field(default_factory=list)
    final_stats: FinalStats = field(default_factory=FinalStats)
    court_schema: CourtSchema | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_video(cls, video_reader: VideoReader, sport: str) -> Self:
        frames = cast(list[FrameT], [cls.frame_class(i) for i in range(video_reader.frame_count)])
        return cls(sport=sport, video=video_reader.info(), frames=frames)

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        frames = cast(list[FrameT], [cls.frame_class.from_dict(f) for f in data["frames"]])
        final_stats = cls.final_stats_class.from_dict(data.get("final_stats") or {})
        return cls(sport=data["sport"], video=data["video"], frames=frames, final_stats=final_stats)

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "video": self.video,
            "frames": [f.to_dict() for f in self.frames],
            "final_stats": self.final_stats.to_dict(),
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, default=_json_default)


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
