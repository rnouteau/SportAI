"""Court evaluator: the 14 court keypoints on still images (YOLO-pose labels).

Metrics per image: Precision / Recall / F1 of the keypoints matched by id, PCK at each
``pck_thresholds`` (share of visible keypoints closer than ``threshold * box size``) and
OKS. TOTAL: Precision / Recall / F1 from the summed counts, PCK and OKS averaged over
the images.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.models.tennis.tennis_court_model import TennisCourtModel
from pipeline.objects.tennis.tennis_court_schema import N_POINTS
from tests.evaluation.base import Evaluator, SampleResult
from tests.evaluation.metrics import Counts

N_KEYPOINTS = N_POINTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def imread_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_yolo_pose_label(
    label_path: Path,
    image_width: int,
    image_height: int,
    n_keypoints: int = N_KEYPOINTS,
) -> tuple[tuple[float, float, float, float], list[tuple[float, float, int]]]:
    """
    Parse one YOLO-pose line.

    Returns (bbox_xywh_abs, keypoints) where each keypoint is (x, y, v) in pixels.
    """
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty label: {label_path}")
    parts = text.split()
    expected = 5 + n_keypoints * 3
    if len(parts) < expected:
        raise ValueError(f"Label too short in {label_path}: {len(parts)} < {expected}")

    values = [float(x) for x in parts[:expected]]
    _cls, xc, yc, bw, bh = values[:5]
    box_w = bw * image_width
    box_h = bh * image_height
    box_x = xc * image_width
    box_y = yc * image_height

    keypoints: list[tuple[float, float, int]] = []
    cursor = 5
    for _ in range(n_keypoints):
        x_n, y_n, v = values[cursor], values[cursor + 1], values[cursor + 2]
        cursor += 3
        keypoints.append((x_n * image_width, y_n * image_height, int(v)))
    return (box_x, box_y, box_w, box_h), keypoints


def bbox_scale(box_w: float, box_h: float) -> float:
    return float(math.sqrt(max(box_w, 0.0) * max(box_h, 0.0)))


def keypoint_distance(
    pred: tuple[float, float] | list[float] | None,
    gt_xy: tuple[float, float],
) -> float:
    if pred is None or len(pred) < 2:
        return float("inf")
    return float(math.hypot(float(pred[0]) - gt_xy[0], float(pred[1]) - gt_xy[1]))


def pck_score(
    preds: list[tuple[float, float] | list[float] | None],
    gts: list[tuple[float, float, int]],
    scale: float,
    threshold: float,
) -> float:
    """Fraction of visible GT keypoints within ``threshold * scale`` of the prediction."""
    if scale <= 0:
        return 0.0
    thr_px = threshold * scale
    visible = 0
    correct = 0
    for pid, (gx, gy, v) in enumerate(gts):
        if v <= 0:
            continue
        visible += 1
        pred = preds[pid] if pid < len(preds) else None
        if keypoint_distance(pred, (gx, gy)) <= thr_px:
            correct += 1
    if visible == 0:
        return 1.0
    return float(correct / visible)


def oks_score(
    preds: list[tuple[float, float] | list[float] | None],
    gts: list[tuple[float, float, int]],
    scale: float,
    sigma: float,
) -> float:
    """Object Keypoint Similarity with uniform sigma for all court corners."""
    if scale <= 0 or sigma <= 0:
        return 0.0
    denom = 0.0
    numer = 0.0
    var = 2.0 * (scale * sigma) ** 2
    for pid, (gx, gy, v) in enumerate(gts):
        if v <= 0:
            continue
        denom += 1.0
        pred = preds[pid] if pid < len(preds) else None
        d2 = keypoint_distance(pred, (gx, gy)) ** 2
        numer += math.exp(-d2 / var)
    if denom == 0.0:
        return 1.0
    return float(numer / denom)


def match_keypoints_by_id(
    preds: list[tuple[float, float] | list[float] | None],
    gts: list[tuple[float, float, int]],
    scale: float,
    threshold: float,
) -> Counts:
    """TP / FP / FN per keypoint id (keypoint ``i`` is only compared with annotation ``i``).

    - visible annotation, prediction within ``threshold * scale`` → TP
    - visible annotation, no prediction                          → FN
    - visible annotation, prediction too far                     → FP + FN
    - invisible annotation, prediction present                   → FP
    """
    thr_px = threshold * scale if scale > 0 else 0.0
    tp = fp = fn = 0
    n = max(len(gts), len(preds), N_KEYPOINTS)
    for pid in range(n):
        gx, gy, v = gts[pid] if pid < len(gts) else (0.0, 0.0, 0)
        pred = preds[pid] if pid < len(preds) else None
        has_pred = pred is not None and len(pred) >= 2
        if v <= 0:
            if has_pred:
                fp += 1
            continue
        if not has_pred:
            fn += 1
            continue
        if keypoint_distance(pred, (gx, gy)) <= thr_px:
            tp += 1
        else:
            fp += 1
            fn += 1
    return Counts(tp=tp, fp=fp, fn=fn)


def discover_image_label_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Support flat ``dir/img.jpg``+``dir/img.txt`` or ``dir/images`` + ``dir/labels``."""
    directory = Path(directory)
    pairs: list[tuple[Path, Path]] = []

    images_dir = directory / "images"
    labels_dir = directory / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
        return pairs

    for image_path in sorted(directory.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def predict_keypoints(
    model: TennisCourtModel,
    image_bgr: np.ndarray,
) -> list[list[float]] | None:
    """Best-instance keypoints ``[[x,y], ...]`` in image pixels (same preprocessing as the pipeline)."""
    prediction = model.predict_keypoints(image_bgr)
    if prediction is None:
        return None
    return prediction[0]


class CourtEvaluator(Evaluator):
    def __init__(self, config: dict, directory: Path) -> None:
        super().__init__(config, directory)
        self.pck_thresholds = [float(t) for t in self.evaluation.get("pck_thresholds") or [0.05]]
        self.match_threshold = float(self.evaluation.get("match_threshold", self.pck_thresholds[0]))
        self.oks_sigma = float(self.evaluation.get("oks_sigma", 0.05))
        self.pck_names = [f"PCK@{t:g}" for t in self.pck_thresholds]
        self.columns = ("Precision", "Recall", "F1-Score", "OKS", *self.pck_names)
        self._model: TennisCourtModel | None = None

    @property
    def model(self) -> TennisCourtModel:
        if self._model is None:
            self._model = TennisCourtModel(self.config["models"][0]["config"])
        return self._model

    def samples(self) -> Iterator[tuple[str, tuple[Path, Path]]]:
        pairs = discover_image_label_pairs(self.directory)
        if not pairs:
            logger.warning(f"No image/label pairs found under {self.directory}")
        for image_path, label_path in pairs:
            yield image_path.name, (image_path, label_path)

    def evaluate(self, sample: tuple[Path, Path]) -> SampleResult | None:
        image_path, label_path = sample
        image = imread_bgr(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        try:
            (_bx, _by, box_w, box_h), gts = parse_yolo_pose_label(label_path, width, height)
        except ValueError as exc:
            logger.warning(f"{exc}")
            return None
        scale = bbox_scale(box_w, box_h)

        pred_pts = predict_keypoints(self.model, image)
        preds: list[tuple[float, float] | list[float] | None] = list(pred_pts or [])
        preds += [None] * (N_KEYPOINTS - len(preds))

        counts = match_keypoints_by_id(preds, gts, scale, self.match_threshold)
        scores = {
            name: pck_score(preds, gts, scale, thr)
            for name, thr in zip(self.pck_names, self.pck_thresholds, strict=True)
        }
        scores["OKS"] = oks_score(preds, gts, scale, self.oks_sigma)
        return SampleResult(self.row(counts, scores), (counts, scores))

    def total(self, results: list[SampleResult]) -> dict:
        counts = sum((r.extra[0] for r in results), Counts())
        names = [*self.pck_names, "OKS"]
        means = {name: float(np.mean([r.extra[1][name] for r in results])) for name in names}
        return self.row(counts, means)

    @staticmethod
    def row(counts: Counts, scores: dict[str, float]) -> dict:
        return {"TP": counts.tp, "FP": counts.fp, "FN": counts.fn, **counts.scores(), **scores}
