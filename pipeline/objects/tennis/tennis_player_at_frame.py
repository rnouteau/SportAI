from dataclasses import dataclass
from typing import Self

from pipeline.objects.player_at_frame import PlayerAtFrame

NEAR, FAR = "near", "far"
SIDES = (NEAR, FAR)


@dataclass
class TennisPlayerAtFrame(PlayerAtFrame):
    side: str | None = None

    def __post_init__(self) -> None:
        if self.side is not None and self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        player = super().from_dict(data)
        player.side = data.get("side")
        return player

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.side is not None:
            d["side"] = self.side
        return d
