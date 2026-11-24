"""Shapes of the API responses (Pydantic). The frontend types are generated from them
(``npx openapi-typescript http://localhost:8000/openapi.json -o src/api/types.ts``)."""

from pydantic import BaseModel


class PlayerSummary(BaseModel):
    id: int
    name: str | None = None
    distance_m: float | None = None
    mean_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    heatmap_url: str | None = None  # None when the heatmap was not rendered


class AnalysisSummary(BaseModel):
    id: str  # folder name under the outputs root, also used in the URLs
    video_name: str
    fps: float
    frame_count: int
    duration_s: float
    overlay_url: str | None = None  # None when the overlay was not rendered
    players: list[PlayerSummary]
