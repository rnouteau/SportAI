from typing import Generic, TypeVar

from pipeline.objects.player import Player

PlayerT = TypeVar("PlayerT", bound=Player)


class FinalStats(Generic[PlayerT]):
    """Whole-video results, stored under ``analysis.data["final_stats"]``."""

    player_class: type[PlayerT]

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def empty(cls) -> dict:
        return {"players": []}

    @property
    def players(self) -> list[PlayerT]:
        return [self.player_class(data) for data in self.data.get("players") or []]

    @players.setter
    def players(self, players: list[PlayerT] | None) -> None:
        self.data["players"] = [player.data for player in (players or [])]
