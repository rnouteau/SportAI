"""Tennis court keypoint schema (14 ITF keypoints) for annotation and homography overlay.

Order matches the Ultralytics Tennis Court Keypoint Detection dataset:
far baseline → far service → near service → near baseline (left→right within each row).
"""

from __future__ import annotations

from dataclasses import dataclass

# ITF dimensions (metres), origin at net centre; +X right, +Y far baseline.
L = 11.885  # half length (total 23.77)
W_D = 5.485  # doubles half-width (total 10.97)
W_S = 4.115  # singles half-width (total 8.23)
S = 6.40  # service line distance from net


@dataclass(frozen=True)
class CourtPoint:
    id: int
    name: str
    x: float
    y: float
    label: str


# 14 keypoints (no net centre). Far = top of broadcast view, near = camera side.
POINTS: tuple[CourtPoint, ...] = (
    CourtPoint(0, "TL_DL", -W_D, L, "Baseline far × doubles left"),
    CourtPoint(1, "TL_SL", -W_S, L, "Baseline far × singles left"),
    CourtPoint(2, "TL_SR", W_S, L, "Baseline far × singles right"),
    CourtPoint(3, "TL_DR", W_D, L, "Baseline far × doubles right"),
    CourtPoint(4, "ST_SL", -W_S, S, "Service far × singles left"),
    CourtPoint(5, "ST_T", 0.0, S, "Service far × centre (T)"),
    CourtPoint(6, "ST_SR", W_S, S, "Service far × singles right"),
    CourtPoint(7, "SV_SL", -W_S, -S, "Service near × singles left"),
    CourtPoint(8, "SV_T", 0.0, -S, "Service near × centre (T)"),
    CourtPoint(9, "SV_SR", W_S, -S, "Service near × singles right"),
    CourtPoint(10, "BL_DL", -W_D, -L, "Baseline near × doubles left"),
    CourtPoint(11, "BL_SL", -W_S, -L, "Baseline near × singles left"),
    CourtPoint(12, "BL_SR", W_S, -L, "Baseline near × singles right"),
    CourtPoint(13, "BL_DR", W_D, -L, "Baseline near × doubles right"),
)

POINT_BY_ID: dict[int, CourtPoint] = {p.id: p for p in POINTS}
POINT_BY_NAME: dict[str, CourtPoint] = {p.name: p for p in POINTS}
N_POINTS = len(POINTS)

# Left-right swap indices for Ultralytics fliplr augmentation.
FLIP_IDX: list[int] = [3, 2, 1, 0, 6, 5, 4, 9, 8, 7, 13, 12, 11, 10]

# Edges between annotated keypoints (drawn on canvas when both ends exist).
EDGES: tuple[tuple[int, int], ...] = (
    # Far baseline
    (0, 1),
    (1, 2),
    (2, 3),
    # Near baseline
    (10, 11),
    (11, 12),
    (12, 13),
    # Doubles sidelines
    (0, 10),
    (3, 13),
    # Singles sidelines (via service)
    (1, 4),
    (4, 7),
    (7, 11),
    (2, 6),
    (6, 9),
    (9, 12),
    # Service lines
    (4, 5),
    (5, 6),
    (7, 8),
    (8, 9),
    # Centre line between the two T's (crosses the net)
    (5, 8),
)

# Extra model-only polylines for overlay (net across singles width; no NET_C keypoint).
OVERLAY_POLYLINES_M: tuple[tuple[tuple[float, float], ...], ...] = (
    ((-W_S, 0.0), (W_S, 0.0)),
)


def model_xy(point_id: int) -> tuple[float, float]:
    p = POINT_BY_ID[point_id]
    return (p.x, p.y)


def overlay_segments_m() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """All court segments in model metres for projection overlay."""
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for a, b in EDGES:
        segments.append((model_xy(a), model_xy(b)))
    for poly in OVERLAY_POLYLINES_M:
        for i in range(len(poly) - 1):
            segments.append((poly[i], poly[i + 1]))
    return segments
