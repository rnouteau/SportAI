from dataclasses import dataclass
from typing import Self

from pipeline.objects.position import CourtPosition, ImagePosition


@dataclass
class PlayerAtFrame:
    bbox: list[float] | None = None
    confidence: float | None = None
    track_id: int | None = None
    player_id: int | None = None
    court: CourtPosition | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        # "position" = key used before the rename to "bbox" (old analysis JSON files)
        bbox = data.get("bbox", data.get("position"))
        court = (data.get("legs") or {}).get("court")
        return cls(
            bbox=bbox,
            confidence=data.get("confidence"),
            track_id=data.get("track_id"),
            player_id=data.get("player_id"),
            court=CourtPosition.from_dict(court) if court else None,
        )

    def to_dict(self) -> dict:
        """Keys whose value is None are left out (the evaluators rely on it)."""
        d: dict = {}
        if self.bbox is not None:
            d["bbox"] = [float(v) for v in self.bbox]
        if self.confidence is not None:
            d["confidence"] = float(self.confidence)
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        if self.player_id is not None:
            d["player_id"] = int(self.player_id)
        if self.court is not None:
            d["legs"] = {"court": self.court.to_dict()}
        return d

    @property
    def feet(self) -> ImagePosition | None:
        if self.bbox is None or len(self.bbox) < 4:
            return None
        x0, _y0, x1, y1 = (float(v) for v in self.bbox[:4])
        return ImagePosition(x=(x0 + x1) / 2.0, y=y1)
