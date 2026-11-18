from typing import Self


class Ball:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        """Build from free kwargs; new fields use setters instead of widening create."""
        return cls(dict(fields))

    @property
    def position(self) -> list[float] | None:
        return self.data.get("position")

    @position.setter
    def position(self, value: list[float] | tuple[float, ...] | None) -> None:
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
