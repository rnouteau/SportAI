from __future__ import annotations

import math
from typing import Self


class CourtPosition:
    """Position on the court plane in metres (JSON-backed)."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        return cls(dict(fields))

    @property
    def x(self) -> float | None:
        value = self.data.get("x")
        return float(value) if value is not None else None

    @x.setter
    def x(self, value: float | None) -> None:
        if value is None:
            self.data.pop("x", None)
        else:
            self.data["x"] = float(value)

    @property
    def y(self) -> float | None:
        value = self.data.get("y")
        return float(value) if value is not None else None

    @y.setter
    def y(self, value: float | None) -> None:
        if value is None:
            self.data.pop("y", None)
        else:
            self.data["y"] = float(value)

    @property
    def distance(self) -> float:
        """Metres from court origin (net centre for tennis)."""
        return math.hypot(float(self.x or 0.0), float(self.y or 0.0))


class ImagePosition:
    """Pixel position (not JSON-backed; computed views like legs)."""

    def __init__(self, x: float, y: float) -> None:
        self._x = float(x)
        self._y = float(y)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y
