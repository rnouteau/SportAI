from dataclasses import dataclass
from typing import Self


@dataclass
class Ball:
    position: list[float]
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(position=data["position"], confidence=data.get("confidence"))

    def to_dict(self) -> dict:
        d: dict = {"position": [float(v) for v in self.position]}
        if self.confidence is not None:
            d["confidence"] = float(self.confidence)
        return d
