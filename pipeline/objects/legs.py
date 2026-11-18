from __future__ import annotations

from pipeline.objects.position import CourtPosition, ImagePosition


class Legs:
    """View: image = bbox bottom-centre (computed); court = persisted CourtPosition."""

    def __init__(self, player_data: dict, bbox: list[float] | None) -> None:
        self._data = player_data
        self._bbox = bbox

    @property
    def image(self) -> ImagePosition | None:
        if self._bbox is None or len(self._bbox) < 4:
            return None
        x0, _y0, x1, y1 = (float(v) for v in self._bbox[:4])
        return ImagePosition((x0 + x1) / 2.0, y1)

    @property
    def x(self) -> float | None:
        image = self.image
        return image.x if image is not None else None

    @property
    def y(self) -> float | None:
        image = self.image
        return image.y if image is not None else None

    @property
    def court(self) -> CourtPosition | None:
        legs_data = self._data.get("legs")
        if not isinstance(legs_data, dict):
            return None
        court_data = legs_data.get("court")
        if not isinstance(court_data, dict):
            return None
        return CourtPosition(court_data)

    @court.setter
    def court(self, value: CourtPosition | None) -> None:
        if value is None:
            legs_data = self._data.get("legs")
            if isinstance(legs_data, dict):
                legs_data.pop("court", None)
                if not legs_data:
                    self._data.pop("legs", None)
            return
        legs_data = self._data.setdefault("legs", {})
        if not isinstance(legs_data, dict):
            legs_data = {}
            self._data["legs"] = legs_data
        legs_data["court"] = value.data
