from dataclasses import dataclass, field
from typing import Self

from pipeline.objects.player import Player


@dataclass
class TennisPlayer(Player):
    """Tennis player over the whole video, with the tracker ids that followed him."""

    track_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        player = super().from_dict(data)
        player.track_ids = [int(v) for v in data.get("track_ids") or []]
        return player

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["track_ids"] = [int(v) for v in self.track_ids]
        return d
