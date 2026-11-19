"""Sport-agnostic court geometry + homography helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np


class CourtSchema(ABC):
    """Shared image↔court projection API; subclasses define court keypoints in metres."""

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Number of annotated court keypoints."""

    @abstractmethod
    def model_xy(self, point_id: int) -> tuple[float, float]:
        """Court-plane coordinates (metres) for keypoint ``point_id``."""

    def estimate_homography(
        self,
        keypoints: list[list[float]] | list[tuple[float, float]],
        confidences: list[float] | None = None,
        *,
        min_points: int = 4,
        min_conf: float = 0.3,
        ransac_reproj_threshold: float = 5.0,
    ) -> np.ndarray | None:
        """Estimate H mapping image pixels → court metres.

        RANSAC runs in the court→image direction so ``ransac_reproj_threshold`` is in
        **pixels** (a threshold in the image→court direction would be in metres, and
        5 m rejects nothing). The result is inverted to keep the image→court convention.
        """
        src: list[list[float]] = []
        dst: list[list[float]] = []
        for i, pt in enumerate(keypoints):
            if i >= self.n_points:
                break
            if confidences is not None and float(confidences[i]) < min_conf:
                continue
            if len(pt) < 2:
                continue
            src.append([float(pt[0]), float(pt[1])])
            mx, my = self.model_xy(i)
            dst.append([mx, my])

        if len(src) < min_points:
            return None

        h_court_to_img, _mask = cv2.findHomography(
            np.asarray(dst, dtype=np.float64),
            np.asarray(src, dtype=np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
        )
        if h_court_to_img is None:
            return None
        try:
            h_img_to_court = h_img_to_court = np.linalg.inv(
                np.asarray(h_court_to_img, dtype=np.float64)
            )
        except np.linalg.LinAlgError:
            return None
        if abs(h_img_to_court[2, 2]) > 1e-12:
            h_img_to_court = h_img_to_court / h_img_to_court[2, 2]
        return h_img_to_court

    def estimate_court_to_image(
        self,
        image_points: list[list[float]] | list[tuple[float, float]],
        point_ids: list[int],
        *,
        min_points: int = 4,
        ransac_reproj_threshold: float = 5.0,
    ) -> np.ndarray | None:
        """Estimate H mapping court metres → image pixels (overlays)."""
        if len(image_points) != len(point_ids):
            raise ValueError("image_points and point_ids must have the same length")
        if len(image_points) < min_points:
            return None

        src = [list(self.model_xy(pid)) for pid in point_ids]
        dst = [[float(p[0]), float(p[1])] for p in image_points]

        h_mat, _mask = cv2.findHomography(
            np.asarray(src, dtype=np.float64),
            np.asarray(dst, dtype=np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
        )
        return h_mat

    def image_to_court(
        self,
        h_mat: np.ndarray,
        pts_xy: list[tuple[float, float]] | list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """Project image points through H (image→court). Returns Nx2 metres."""
        arr = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, h_mat).reshape(-1, 2)

    def distance(self, x: float, y: float) -> float:
        """Metres from court origin (subclass defines origin; tennis = net centre)."""
        return math.hypot(float(x), float(y))
