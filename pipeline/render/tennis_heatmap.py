"""Heatmap of the court positions of each final player, saved as one PNG per player.

The court and its surroundings are cut into ``CELL_M`` cells; each cell counts the frames
the player spent in it (``np.histogram2d`` on ``player.court``). The counts are blurred,
coloured and the court lines are drawn on top. The image is seen from above with the
far side (+y) at the top: the near player (P1) is in the bottom half.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_court_schema import W_D, L, TennisCourtSchema

CELL_M = 0.25  # size of a cell (metres)
MARGIN_X_M = 4.0  # beyond the doubles sidelines
MARGIN_Y_M = 8.0  # behind the baselines (players often stand far back)
PX_PER_M = 20  # image scale
BLUR_M = 0.5  # gaussian blur, for a smooth spot instead of a mosaic
HEADER_PX = 40  # band above the court for the title

HALF_WIDTH_M = W_D + MARGIN_X_M
HALF_LENGTH_M = L + MARGIN_Y_M

LINE_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)


def player_positions(analysis: TennisAnalysis, player_id: int) -> list[tuple[float, float]]:
    """Court positions ``(x, y)`` in metres of one player, frames without a position skipped."""
    return [
        (player.court.x, player.court.y)
        for frame in analysis.frames
        for player in frame.players
        if player.player_id == player_id and player.court is not None
    ]


def position_histogram(
    positions: list[tuple[float, float]],
    half_width_m: float = HALF_WIDTH_M,
    half_length_m: float = HALF_LENGTH_M,
    cell_m: float = CELL_M,
) -> np.ndarray:
    """Frames per cell, as an image-oriented grid: row 0 = far end (+y), column 0 = left (-x).

    Positions outside the area are not counted (logged: many of them = a bad homography).
    """
    n_cols = round(2 * half_width_m / cell_m)
    n_rows = round(2 * half_length_m / cell_m)
    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)
    counts, _, _ = np.histogram2d(
        xs,
        ys,
        bins=(n_cols, n_rows),
        range=((-half_width_m, half_width_m), (-half_length_m, half_length_m)),
    )
    outside = len(positions) - int(counts.sum())
    if outside:
        logger.warning(f"{outside} position(s) outside the heatmap area, not counted")
    # histogram2d gives [x, y]; an image is [row, column] read from the top.
    return np.flipud(counts.T)


def court_to_pixel(
    x: float,
    y: float,
    half_width_m: float = HALF_WIDTH_M,
    half_length_m: float = HALF_LENGTH_M,
) -> tuple[int, int]:
    """Court metres → heatmap pixel (column, row), same orientation as ``position_histogram``."""
    return (round((x + half_width_m) * PX_PER_M), round((half_length_m - y) * PX_PER_M))


def render_heatmap(
    histogram: np.ndarray,
    title: str = "",
    half_width_m: float = HALF_WIDTH_M,
    half_length_m: float = HALF_LENGTH_M,
    cell_m: float = CELL_M,
) -> np.ndarray:
    """BGR image of a ``position_histogram``: title band, then the court seen from above."""
    width = round(2 * half_width_m * PX_PER_M)
    height = round(2 * half_length_m * PX_PER_M)

    sigma = BLUR_M / cell_m  # in cells
    heat = cv2.GaussianBlur(histogram.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    peak = float(heat.max())
    if peak > 0:  # an empty histogram stays black
        heat = np.asarray(heat / peak, dtype=np.float32)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    image = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    for a, b in TennisCourtSchema().overlay_segments_m():
        cv2.line(
            image,
            court_to_pixel(*a, half_width_m, half_length_m),
            court_to_pixel(*b, half_width_m, half_length_m),
            LINE_COLOR,
            2,
            cv2.LINE_AA,
        )
    header = np.zeros((HEADER_PX, width, 3), dtype=np.uint8)
    if title:
        cv2.putText(
            header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA
        )
    return np.vstack([header, image])


def render_heatmaps(analysis: TennisAnalysis, out_dir: str | Path) -> list[Path]:
    """One ``heatmap_player_<id>.png`` per final player in ``out_dir``; returns the files written."""
    players = analysis.final_stats.players
    if not players:
        logger.warning("No final players (TennisPlayerSelectorStat missing?): no heatmap")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float((analysis.video or {}).get("fps") or 0.0)

    written: list[Path] = []
    for player in players:
        positions = player_positions(analysis, player.id)
        seconds = f" - {len(positions) / fps:.1f} s" if fps > 0 else ""
        title = f"{player.name or f'P{player.id}'}{seconds}"
        image = render_heatmap(position_histogram(positions), title)
        path = out_dir / f"heatmap_player_{player.id}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        logger.info(f"Heatmap of player {player.id} ({len(positions)} positions): {path}")
        written.append(path)
    return written
