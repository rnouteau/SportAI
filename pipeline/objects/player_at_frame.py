from typing import Self

from pipeline.objects.legs import Legs


class PlayerAtFrame:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        """Build from free kwargs; new fields use setters (e.g. legs.court)."""
        return cls(dict(fields))

    @property
    def track_id(self) -> int | None:
        """Tracker id (stable across frames); None when the tracker gave no id."""
        value = self.data.get("track_id")
        return int(value) if value is not None else None

    @track_id.setter
    def track_id(self, value: int | None) -> None:
        if value is None:
            self.data.pop("track_id", None)
        else:
            self.data["track_id"] = int(value)

    @property
    def player_id(self) -> int | None:
        """Stable player id set by the player selector (1 = near, 2 = far)."""
        value = self.data.get("player_id")
        return int(value) if value is not None else None

    @player_id.setter
    def player_id(self, value: int | None) -> None:
        if value is None:
            self.data.pop("player_id", None)
        else:
            self.data["player_id"] = int(value)

    @property
    def bbox(self) -> list[float] | None:
        return self.data.get("position")

    @bbox.setter
    def bbox(self, value: list[float] | tuple[float, ...] | None) -> None:
        if value is None:
            self.data.pop("position", None)
        else:
            self.data["position"] = [float(x) for x in value]

    @property
    def confidence(self) -> float | None:
        value = self.data.get("confidence")
        return float(value) if value is not None else None

    @confidence.setter
    def confidence(self, value: float | None) -> None:
        if value is None:
            self.data.pop("confidence", None)
        else:
            self.data["confidence"] = float(value)

    @property
    def legs(self) -> Legs:
        """Legs view: image computed from bbox; court persisted under data['legs']['court']."""
        return Legs(self.data, self.bbox)
