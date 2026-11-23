"""Player heatmaps: histogram orientation, metres → pixels, images written per player."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_player import TennisPlayer
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.render.tennis_heatmap import (
    HALF_LENGTH_M,
    HALF_WIDTH_M,
    HEADER_PX,
    PX_PER_M,
    court_to_pixel,
    player_positions,
    position_histogram,
    render_heatmap,
    render_heatmaps,
)
from tests.helpers import make_tennis_analysis

IMAGE_SHAPE = (
    HEADER_PX + round(2 * HALF_LENGTH_M * PX_PER_M),
    round(2 * HALF_WIDTH_M * PX_PER_M),
    3,
)


def test_one_position_at_the_centre() -> None:
    grid = position_histogram([(0.01, 0.01)], half_width_m=2.0, half_length_m=3.0, cell_m=1.0)
    assert grid.shape == (6, 4)  # rows = length, columns = width
    assert grid.sum() == 1
    assert grid[2, 2] == 1  # just right of and above the centre


def test_far_side_is_at_the_top_and_left_is_on_the_left() -> None:
    grid = position_histogram([(-1.5, 10.0), (1.5, -10.0)])
    rows, cols = grid.shape
    far_row, far_col = np.argwhere(grid)[0]  # first nonzero cell = top of the image
    near_row, near_col = np.argwhere(grid)[1]
    assert far_row < rows / 2 < near_row
    assert far_col < cols / 2 < near_col


def test_positions_outside_the_area_are_not_counted() -> None:
    assert position_histogram([(50.0, 0.0), (0.0, 0.0)]).sum() == 1
    assert position_histogram([]).sum() == 0


def test_court_to_pixel() -> None:
    assert court_to_pixel(-HALF_WIDTH_M, HALF_LENGTH_M) == (0, 0)
    assert court_to_pixel(0.0, 0.0) == (
        round(HALF_WIDTH_M * PX_PER_M),
        round(HALF_LENGTH_M * PX_PER_M),
    )


def test_render_heatmap_image_even_when_empty() -> None:
    for positions in ([], [(0.0, -11.0)] * 50):
        image = render_heatmap(position_histogram(positions), "P1")
        assert image.shape == IMAGE_SHAPE and image.dtype == np.uint8


def test_hot_spot_where_the_player_stands() -> None:
    image = render_heatmap(position_histogram([(0.0, -11.0)] * 50))
    col, row = court_to_pixel(0.0, -11.0)
    row += HEADER_PX  # the title band is above the court
    hot = image[row - 3 : row + 4, col + 5 : col + 9].astype(int).sum()  # beside the centre line
    cold = image[row - 3 : row + 4, 10:14].astype(int).sum()  # far on the left
    assert hot > cold


def test_render_heatmaps_writes_one_png_per_final_player(tmp_path: Path) -> None:
    analysis = make_tennis_analysis(n_frames=10, fps=10.0)
    for frame in analysis.frames:
        frame.players = [
            TennisPlayerAtFrame(player_id=1, court=CourtPosition(x=1.0, y=-11.0)),
            TennisPlayerAtFrame(player_id=2, court=CourtPosition(x=-1.0, y=11.0)),
            TennisPlayerAtFrame(player_id=None, court=CourtPosition(x=0.0, y=0.0)),
        ]
    analysis.final_stats.players = [TennisPlayer(id=1, name="player A"), TennisPlayer(id=2)]

    assert len(player_positions(analysis, 1)) == 10
    written = render_heatmaps(analysis, tmp_path / "heatmaps")

    assert [p.name for p in written] == ["heatmap_player_1.png", "heatmap_player_2.png"]
    image = cv2.imread(str(written[0]))
    assert image is not None and image.shape == IMAGE_SHAPE


def test_no_final_players_no_heatmap(tmp_path: Path) -> None:
    assert render_heatmaps(make_tennis_analysis(), tmp_path) == []
