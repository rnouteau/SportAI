from dataclasses import dataclass, field
from typing import Self

from pipeline.objects.player import Player


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


@dataclass
class TennisPlayer(Player):
    """Tennis player over the whole video: tracker ids that followed him and his movement."""

    track_ids: list[int] = field(default_factory=list)
    distance_m: float | None = None  # TennisPlayerMovementStat
    mean_speed_kmh: float | None = None
    max_speed_kmh: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        player = super().from_dict(data)
        player.track_ids = [int(v) for v in data.get("track_ids") or []]
        player.distance_m = _optional_float(data.get("distance_m"))
        player.mean_speed_kmh = _optional_float(data.get("mean_speed_kmh"))
        player.max_speed_kmh = _optional_float(data.get("max_speed_kmh"))
        return player

    def to_dict(self) -> dict:
        """Movement fields are written only when the stat ran (no ``null``)."""
        d = super().to_dict()
        d["track_ids"] = [int(v) for v in self.track_ids]
        for key in ("distance_m", "mean_speed_kmh", "max_speed_kmh"):
            value = getattr(self, key)
            if value is not None:
                d[key] = float(value)
        return d
