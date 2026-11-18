"""Temporal smoothing of per-frame keypoints (sport-agnostic)."""

from __future__ import annotations

from collections import deque

import numpy as np


class KeypointSmoother:
    """Sliding-window median over the last ``window`` detections, per keypoint.

    - Only keypoints with ``conf >= min_conf`` contribute to the median.
    - If the confident keypoints of a new detection moved more than ``reset_px``
      (median displacement vs the current smoothed estimate), the buffer is cleared:
      this is a scene cut or a camera move, and old frames must not be mixed in.
    - ``window <= 1`` disables smoothing (pass-through).
    """

    def __init__(self, window: int = 5, min_conf: float = 0.3, reset_px: float | None = None):
        self.window = max(int(window), 1)
        self.min_conf = float(min_conf)
        self.reset_px = reset_px
        self._xy: deque[np.ndarray] = deque(maxlen=self.window)
        self._conf: deque[np.ndarray] = deque(maxlen=self.window)
        self._last: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self._xy.clear()
        self._conf.clear()
        self._last = None

    def update(
        self,
        keypoints: list[list[float]] | np.ndarray,
        confidences: list[float] | np.ndarray,
    ) -> tuple[list[list[float]], list[float]]:
        xy = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
        if xy.shape[0] != conf.shape[0]:
            raise ValueError("keypoints and confidences must have the same length")

        if self.window <= 1:
            return xy.tolist(), conf.tolist()

        if self._should_reset(xy, conf):
            self.reset()

        self._xy.append(xy)
        self._conf.append(conf)

        stack_xy = np.stack(self._xy)  # (T, K, 2)
        stack_conf = np.stack(self._conf)  # (T, K)
        valid = stack_conf >= self.min_conf

        out_xy = xy.copy()
        out_conf = conf.copy()
        for k in range(xy.shape[0]):
            mask = valid[:, k]
            if not mask.any():
                continue
            out_xy[k] = np.median(stack_xy[mask, k], axis=0)
            out_conf[k] = float(np.median(stack_conf[mask, k]))

        self._last = (out_xy, out_conf)
        return out_xy.tolist(), out_conf.tolist()

    def _should_reset(self, xy: np.ndarray, conf: np.ndarray) -> bool:
        if self.reset_px is None or self._last is None:
            return False
        last_xy, last_conf = self._last
        if last_xy.shape != xy.shape:
            return True
        both = (conf >= self.min_conf) & (last_conf >= self.min_conf)
        if not both.any():
            return False
        moved = np.linalg.norm(xy[both] - last_xy[both], axis=1)
        return bool(np.median(moved) > self.reset_px)
