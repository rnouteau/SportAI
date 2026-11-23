from dataclasses import dataclass
from typing import Self


@dataclass
class Player:
    """A player over the whole video, listed in ``analysis.final_stats.players``."""

    id: int
    name: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(id=int(data["id"]), name=data.get("name"))

    def to_dict(self) -> dict:
        d: dict = {"id": int(self.id)}
        if self.name is not None:
            d["name"] = self.name
        return d
