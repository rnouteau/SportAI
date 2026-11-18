"""OpenCV homography helpers: image pixels <-> tennis court metres."""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.objects.tennis.tennis_court_schema import N_POINTS, model_xy


def estimate_image_to_court(
    keypoints: list[list[float]] | list[tuple[float, float]],
    confidences: list[float] | None = None,
    *,
    min_points: int = 4,
    min_conf: float = 0.3,
    ransac_reproj_threshold: float = 5.0,
) -> np.ndarray | None:
    """Estimate H mapping image pixels → court metres (origin at net centre)."""
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for i, pt in enumerate(keypoints):
        if i >= N_POINTS:
            break
        if confidences is not None and float(confidences[i]) < min_conf:
            continue
        if len(pt) < 2:
            continue
        src.append([float(pt[0]), float(pt[1])])
        mx, my = model_xy(i)
        dst.append([mx, my])

    if len(src) < min_points:
        return None

    h_mat, _mask = cv2.findHomography(
        np.asarray(src, dtype=np.float64),
        np.asarray(dst, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    return h_mat


def estimate_court_to_image(
    image_points: list[list[float]] | list[tuple[float, float]],
    point_ids: list[int],
    *,
    min_points: int = 4,
    ransac_reproj_threshold: float = 5.0,
) -> np.ndarray | None:
    """Estimate H mapping court metres → image pixels (for overlays)."""
    if len(image_points) != len(point_ids):
        raise ValueError("image_points and point_ids must have the same length")
    if len(image_points) < min_points:
        return None

    src = [list(model_xy(pid)) for pid in point_ids]
    dst = [[float(p[0]), float(p[1])] for p in image_points]

    h_mat, _mask = cv2.findHomography(
        np.asarray(src, dtype=np.float64),
        np.asarray(dst, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    return h_mat


def image_to_court(
    h_mat: np.ndarray,
    pts_xy: list[tuple[float, float]] | list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Project image points through H (image→court). Returns Nx2 array in metres."""
    arr = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, h_mat).reshape(-1, 2)
    return out


def distance_to_center_m(court_xy: tuple[float, float] | list[float]) -> float:
    return float(np.hypot(float(court_xy[0]), float(court_xy[1])))
