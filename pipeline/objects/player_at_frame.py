from typing import Self

from pipeline.objects.position import ImagePosition


class PlayerAtFrame:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        """Build from free kwargs; new fields use setters (e.g. legs.court later)."""
        return cls(dict(fields))

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
    def legs(self) -> ImagePosition | None:
        """Bottom-centre of bbox (computed; not stored in JSON)."""
        if self.bbox is None or len(self.bbox) < 4:
            return None
        x0, _y0, x1, y1 = (float(v) for v in self.bbox[:4])
        return ImagePosition((x0 + x1) / 2.0, y1)
