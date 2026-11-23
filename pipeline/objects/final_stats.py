from dataclasses import dataclass, field
from typing import ClassVar, Generic, Self, TypeVar, cast

from pipeline.objects.player import Player

PlayerT = TypeVar("PlayerT", bound=Player)


@dataclass
class FinalStats(Generic[PlayerT]):
    """Whole-video results (``analysis.final_stats``), e.g. the players and their ids."""

    player_class: ClassVar[type[Player]] = Player
    players: list[PlayerT] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        players = cast(
            list[PlayerT], [cls.player_class.from_dict(p) for p in data.get("players") or []]
        )
        return cls(players=players)

    def to_dict(self) -> dict:
        return {"players": [p.to_dict() for p in self.players]}
