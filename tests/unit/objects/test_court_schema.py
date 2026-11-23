"""Court schema: homography estimation (image → court metres) and projection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_court_schema import N_POINTS, TennisCourtSchema, model_xy
from tests.helpers import COURT_OFFSET, shifted_court_keypoints


def broadcast_view_keypoints() -> tuple[list[list[float]], np.ndarray]:
    """The 14 court points seen through a realistic broadcast court→image homography."""
    schema = TennisCourtSchema()
    court_corners = np.float32(
        [[-5.485, 11.885], [5.485, 11.885], [-5.485, -11.885], [5.485, -11.885]]
    )
    image_corners = np.float32([[650, 315], [1250, 315], [410, 770], [1495, 775]])
    h_court_to_img = cv2.getPerspectiveTransform(court_corners, image_corners).astype(np.float64)
    model_pts = np.array([schema.model_xy(i) for i in range(N_POINTS)], dtype=np.float64)
    img_pts = cv2.perspectiveTransform(model_pts.reshape(-1, 1, 2), h_court_to_img).reshape(-1, 2)
    return img_pts.tolist(), h_court_to_img


def test_translation_is_recovered() -> None:
    schema = TennisCourtSchema()
    h_mat = schema.estimate_homography(shifted_court_keypoints(), [1.0] * N_POINTS, min_conf=0.3)
    assert h_mat is not None
    mx, my = model_xy(5)
    court = schema.image_to_court(h_mat, [(mx + COURT_OFFSET[0], my + COURT_OFFSET[1])])[0]
    assert court == pytest.approx([mx, my], abs=1e-3)


def test_too_few_confident_points_give_no_homography() -> None:
    keypoints = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (N_POINTS - 3)
    confidences = [1.0, 1.0, 1.0] + [0.0] * (N_POINTS - 3)
    assert TennisCourtSchema().estimate_homography(keypoints, confidences, min_conf=0.5) is None


def test_ransac_rejects_a_keypoint_60_px_off() -> None:
    """The RANSAC threshold is in pixels, not in metres."""
    schema = TennisCourtSchema()
    keypoints, _ = broadcast_view_keypoints()
    keypoints[8] = [keypoints[8][0] + 60.0, keypoints[8][1] - 40.0]  # near T, badly predicted

    h_mat = schema.estimate_homography(keypoints, [1.0] * N_POINTS, ransac_reproj_threshold=5.0)
    assert h_mat is not None
    clean, _ = broadcast_view_keypoints()
    projected = schema.image_to_court(h_mat, clean)
    expected = [model_xy(pid) for pid in range(N_POINTS)]
    assert np.allclose(projected, expected, atol=0.05)


def test_matrix_maps_image_to_court_and_is_normalised() -> None:
    keypoints, h_court_to_img = broadcast_view_keypoints()
    h_mat = TennisCourtSchema().estimate_homography(keypoints, [1.0] * N_POINTS)
    assert h_mat is not None
    assert h_mat[2, 2] == pytest.approx(1.0)
    expected = np.linalg.inv(h_court_to_img)
    assert np.allclose(h_mat, expected / expected[2, 2], atol=1e-6)


def test_court_position_from_image() -> None:
    schema = TennisCourtSchema()
    h_mat = schema.estimate_homography(shifted_court_keypoints(), [1.0] * N_POINTS)
    assert h_mat is not None
    pos = CourtPosition.from_image((3.0 + COURT_OFFSET[0], 4.0 + COURT_OFFSET[1]), h_mat, schema)
    assert (pos.x, pos.y) == pytest.approx((3.0, 4.0), abs=1e-2)
    assert pos.distance == pytest.approx(5.0, abs=1e-2)


def test_tennis_schema() -> None:
    assert TennisCourtSchema().n_points == N_POINTS == 14
