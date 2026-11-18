"""Unit tests for court homography helpers and dict-first objects (part 1)."""

from __future__ import annotations

import pytest

from pipeline.objects.player_at_frame import PlayerAtFrame
from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.court_homography import (
    distance_to_center_m,
    estimate_image_to_court,
    image_to_court,
)
from pipeline.objects.tennis.tennis_court_schema import N_POINTS, model_xy
from pipeline.objects.tennis.tennis_homography import TennisHomography


@pytest.mark.unit
def test_estimate_image_to_court_identity_scale() -> None:
    offset = (100.0, 200.0)
    keypoints = []
    confidences = []
    for pid in range(N_POINTS):
        mx, my = model_xy(pid)
        keypoints.append([mx + offset[0], my + offset[1]])
        confidences.append(1.0)

    h_mat = estimate_image_to_court(keypoints, confidences, min_conf=0.3)
    assert h_mat is not None

    pid = 5
    mx, my = model_xy(pid)
    img_pt = (mx + offset[0], my + offset[1])
    court = image_to_court(h_mat, [img_pt])[0]
    assert court[0] == pytest.approx(mx, abs=1e-3)
    assert court[1] == pytest.approx(my, abs=1e-3)


@pytest.mark.unit
def test_estimate_image_to_court_too_few_points() -> None:
    keypoints = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (N_POINTS - 3)
    confidences = [1.0, 1.0, 1.0] + [0.0] * (N_POINTS - 3)
    assert estimate_image_to_court(keypoints, confidences, min_conf=0.5) is None


@pytest.mark.unit
def test_legs_image_bottom_center_computed() -> None:
    player = PlayerAtFrame.create(position=[10.0, 20.0, 30.0, 40.0], confidence=0.9)
    assert player.legs is not None
    assert player.legs.x == pytest.approx(20.0)
    assert player.legs.y == pytest.approx(40.0)
    assert "legs" not in player.data


@pytest.mark.unit
def test_court_position_distance_and_setters() -> None:
    pos = CourtPosition.create(x=3.0, y=4.0)
    assert pos.x == 3.0
    assert pos.y == 4.0
    assert pos.distance == pytest.approx(5.0)
    assert distance_to_center_m([3.0, 4.0]) == pytest.approx(5.0)

    pos2 = CourtPosition({})
    pos2.x = 3.0
    pos2.y = 4.0
    assert pos2.data == {"x": 3.0, "y": 4.0}
    assert pos2.distance == pytest.approx(5.0)


@pytest.mark.unit
def test_homography_matrix_via_setter() -> None:
    homo = TennisHomography.create(
        keypoints=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        confidences=[1.0, 1.0, 1.0, 1.0],
    )
    assert homo.matrix is None
    homo.matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert homo.matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert "matrix" in homo.data
