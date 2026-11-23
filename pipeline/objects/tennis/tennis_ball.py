from dataclasses import dataclass
from typing import Self

from pipeline.objects.ball import Ball


@dataclass
class TennisBall(Ball):
    velocity: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        ball = super().from_dict(data)
        ball.velocity = data.get("velocity")
        return ball

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.velocity is not None:
            d["velocity"] = float(self.velocity)
        return d
