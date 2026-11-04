

from pipeline.objects.ball import Ball
from pipeline.objects.player_at_frame import PlayerAtFrame


class Frame():
    def __init__(self,data : dict):
        self.data = data

    @property
    def id(self) -> int | None:
        return self.data.get("id", None)

    @id.setter
    def id(self, value : int) -> None:
        self.data["id"] = value

