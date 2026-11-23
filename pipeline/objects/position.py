from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from pipeline.objects.court_schema import CourtSchema


@dataclass(slots=True)
class CourtPosition:
    """Position on the court plane in metres (JSON-backed)."""

    x: float
    y: float

    @classmethod
    def from_image(
        cls,
        image_xy: tuple[float, float] | list[float],
        matrix: np.ndarray | list[list[float]],
        schema: CourtSchema,
    ) -> Self:
        """Project image pixels → court metres via schema.image_to_court."""
        h_mat = np.asarray(matrix, dtype=np.float64)
        pt: tuple[float, float] = (float(image_xy[0]), float(image_xy[1]))
        court_xy = schema.image_to_court(h_mat, [pt])[0]
        return cls(x=float(court_xy[0]), y=float(court_xy[1]))

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(x=float(data["x"]), y=float(data["y"]))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @property
    def distance(self) -> float:
        """Metres from court origin (net centre for tennis)."""
        return math.hypot(self.x, self.y)

    def distance_to(self, other: CourtPosition) -> float:
        """Metres between two court positions."""
        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )


@dataclass(slots=True, frozen=True)
class ImagePosition:
    """Pixel position (not JSON-backed; e.g. player.feet, computed from the bbox)."""

    x: float
    y: float
