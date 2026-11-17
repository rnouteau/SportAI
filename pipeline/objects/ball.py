from typing import Self


class Ball:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, position: tuple[float, float] | list[float], confidence: float) -> Self:
        return cls(
            {
                "position": [float(x) for x in position],
                "confidence": float(confidence),
            }
        )

    @property
    def position(self) -> list[float] | None:
        return self.data.get("position", None)

    @property
    def confidence(self) -> float | None:
        return self.data.get("confidence", None)
