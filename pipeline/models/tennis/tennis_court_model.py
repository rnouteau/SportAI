import math
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from pipeline.device import resolve_device
from pipeline.models.model import Model
from pipeline.objects.keypoint_smoother import KeypointSmoother
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.video_reader import VideoReader


def parse_input_size(value: object, default: int = 640) -> tuple[int, int]:
    """Config ``input_size`` → (height, width). Accepts an int or a [h, w] pair."""
    if value is None:
        return default, default
    if isinstance(value, int | float):
        return int(value), int(value)
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"input_size must be an int or [h, w], got {value!r}")


def predict_court_keypoints(
    model: YOLO,
    image: np.ndarray,
    *,
    device: str = "cpu",
    input_size: tuple[int, int] = (640, 640),
    stretch: bool = True,
) -> tuple[list[list[float]], list[float]] | None:
    """Run the court pose model on one BGR frame; keypoints are returned in frame pixels.

    ``stretch=True`` reproduces the training preprocessing: the dataset images were
    resized to 640x640 **without** keeping the aspect ratio (Roboflow "Stretch"), so the
    frame is stretched the same way before inference and keypoints are scaled back.
    With ``stretch=False`` Ultralytics letterboxes the frame (only correct for a model
    trained on letterboxed / aspect-preserving images).
    """
    frame_h, frame_w = image.shape[:2]
    in_h, in_w = input_size

    if stretch:
        source = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = frame_w / in_w, frame_h / in_h
    else:
        source = image
        scale_x = scale_y = 1.0

    results = list(
        model.predict(
            source,
            imgsz=[in_h, in_w],
            classes=[0],
            verbose=False,
            device=device,
        )
    )
    if not results:
        return None
    keypoints = getattr(results[0], "keypoints", None)
    if keypoints is None or len(keypoints) == 0:
        return None
    if keypoints.conf is None or keypoints.xy is None:
        return None

    # Shapes: xy (N, K, 2), conf (N, K) — keep the best instance.
    xy = torch.as_tensor(keypoints.xy).detach().cpu().double()
    conf = torch.as_tensor(keypoints.conf).detach().cpu().double()
    if xy.ndim != 3 or conf.ndim != 2 or xy.shape[0] == 0:
        return None
    best = int(conf.mean(dim=1).argmax().item())
    pts = xy[best].clone()
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    return pts.tolist(), conf[best].tolist()


class TennisCourtModel(Model[TennisAnalysis]):
    """Court keypoints (YOLO pose, 14 kpts) → per-frame homography.

    Config keys:
      weight, device
      input_size: 640            # training imgsz (int or [h, w])
      stretch_input: true        # dataset was exported as stretched 640x640
      min_conf: 0.3              # keypoint confidence for homography / smoothing
      ransac_px: 5.0             # RANSAC reprojection threshold, in pixels
      smoothing_window: 5        # frames in the temporal median (1 = off)
      smoothing_reset_ratio: 0.02  # scene-cut reset, fraction of the frame diagonal
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisCourtModel initialized")
        weight = self.config.get("weight")
        self.weight: str | None = str(weight) if weight else None
        self.device = resolve_device(str(self.config.get("device", "cpu")))
        self.model: YOLO | None = YOLO(self.weight) if self.weight else None
        self.input_size = parse_input_size(self.config.get("input_size", 640))
        self.stretch_input = bool(self.config.get("stretch_input", True))
        self.min_conf = float(self.config.get("min_conf", 0.3))
        self.ransac_px = float(self.config.get("ransac_px", 5.0))
        self.smoothing_window = int(self.config.get("smoothing_window", 5))
        self.smoothing_reset_ratio = float(self.config.get("smoothing_reset_ratio", 0.02))

        if self.model:
            logger.info(f"Model loaded with weight: {self.weight} on device: {self.device}")

    def predict_keypoints(
        self, image: np.ndarray
    ) -> tuple[list[list[float]], list[float]] | None:
        if self.model is None:
            return None
        return predict_court_keypoints(
            self.model,
            image,
            device=self.device,
            input_size=self.input_size,
            stretch=self.stretch_input,
        )

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisCourtModel")

        if self.model is None:
            logger.error("No weight loaded, cannot run")
            return

        schema = analysis.court_schema
        if schema is None:
            logger.error("No court_schema on analysis, cannot estimate homography")
            return

        smoother: KeypointSmoother | None = None
        n_frames = len(analysis.frames)
        for frame_id, image in video_reader.iter_frames():
            if frame_id >= n_frames:
                break

            if smoother is None:
                frame_h, frame_w = image.shape[:2]
                smoother = KeypointSmoother(
                    window=self.smoothing_window,
                    min_conf=self.min_conf,
                    reset_px=self.smoothing_reset_ratio * math.hypot(frame_w, frame_h),
                )

            prediction = self.predict_keypoints(image)
            if prediction is None:
                continue
            pts, confs = smoother.update(*prediction)

            h_mat = schema.estimate_homography(
                pts,
                confs,
                min_conf=self.min_conf,
                ransac_reproj_threshold=self.ransac_px,
            )
            homo = TennisHomography(keypoints=pts, confidences=confs, matrix=h_mat.tolist() if h_mat is not None else None)
            if h_mat is not None:
                homo.matrix = h_mat.tolist()
            analysis.frames[frame_id].homography = homo
