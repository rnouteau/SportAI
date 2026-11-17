from typing import Self


class PlayerAtFrame:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(
        cls, bbox: tuple[float, float, float, float] | list[float], confidence: float
    ) -> Self:
        return cls(
            {
                "position": [float(x) for x in bbox],
                "confidence": float(confidence),
            }
        )

    @property
    def bbox(self) -> list[float] | None:
        return self.data.get("position", None)

    @property
    def confidence(self) -> float | None:
        return self.data.get("confidence", None)
