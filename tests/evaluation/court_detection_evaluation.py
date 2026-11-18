"""Court keypoint evaluator: PCK / OKS / Acc-Prec-Rec-F1 on still images (YOLO-pose labels)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from ultralytics.engine.results import Results

from pipeline.models.tennis.tennis_court_model import TennisCourtModel

N_KEYPOINTS = 14
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
) -> tuple[int, int, int]:
    """
    TP/FP/FN per keypoint id.

    - GT visible + pred within thr → TP
    - GT visible + miss/far → FN (and a FP if a pred exists but is far — counted as FN only for recall;
      far pred on visible GT counts as FP+FN pair → we count FP for wrong pred and FN for missed GT)
    - GT invisible: ignore GT; pred present → FP
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
    return tp, fp, fn


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
    """Run Ultralytics pose and return best-instance keypoints as ``[[x,y], ...]``."""
    if model.model is None:
        return None
    results = model.model.predict(
        image_bgr,
        classes=[0],
        stream=False,
        verbose=False,
        device=model.device,
    )
    if not results:
        return None
    result = results[0]
    if not isinstance(result, Results):
        return None
    keypoints = result.keypoints
    if keypoints is None or len(keypoints) == 0:
        return None
    if keypoints.conf is None or keypoints.xy is None:
        return None
    xy = torch.as_tensor(keypoints.xy).detach().cpu()
    conf = torch.as_tensor(keypoints.conf).detach().cpu()
    if xy.ndim != 3 or conf.ndim != 2 or xy.shape[0] == 0:
        return None
    best = int(conf.mean(dim=1).argmax().item())
    return xy[best].tolist()


class CourtDetectionEvaluator:
    def __init__(self, config: dict, directory: Path) -> None:
        logger.info("Initialized CourtDetectionEvaluator")
        self.config = config
        self.directory = Path(directory)
        evaluation = config.get("evaluation", {})
        self.pck_thresholds = [float(t) for t in evaluation.get("pck_thresholds", [0.05, 0.1])]
        if not self.pck_thresholds:
            self.pck_thresholds = [0.05]
        self.match_threshold = float(evaluation.get("match_threshold", self.pck_thresholds[0]))
        self.oks_sigma = float(evaluation.get("oks_sigma", 0.05))
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0

    def run_test(self) -> list[dict]:
        model_cfg = self.config["models"][0]["config"]
        model = TennisCourtModel(model_cfg)

        pairs = discover_image_label_pairs(self.directory)
        if not pairs:
            logger.warning("No image/label pairs found under {}", self.directory)

        self.results: list[dict] = []
        totals = {"tp": 0, "fp": 0, "fn": 0}
        pck_sums = {thr: 0.0 for thr in self.pck_thresholds}
        oks_sum = 0.0
        n_images = 0

        for image_path, label_path in pairs:
            image = imread_bgr(image_path)
            if image is None:
                logger.warning("Could not read {}, skipped", image_path.name)
                continue
            height, width = image.shape[:2]
            try:
                bbox, gts = parse_yolo_pose_label(label_path, width, height)
            except ValueError as exc:
                logger.warning("{}", exc)
                continue

            _bx, _by, box_w, box_h = bbox
            scale = bbox_scale(box_w, box_h)
            pred_pts = predict_keypoints(model, image)
            preds: list[tuple[float, float] | list[float] | None]
            if pred_pts is None:
                preds = [None] * N_KEYPOINTS
            else:
                preds = list(pred_pts)
                while len(preds) < N_KEYPOINTS:
                    preds.append(None)

            self.reset()
            tp, fp, fn = match_keypoints_by_id(preds, gts, scale, self.match_threshold)
            self.tp, self.fp, self.fn = tp, fp, fn

            metrics = self.compute()
            for thr in self.pck_thresholds:
                metrics[f"PCK@{thr:g}"] = pck_score(preds, gts, scale, thr)
                pck_sums[thr] += metrics[f"PCK@{thr:g}"]
            metrics["OKS"] = oks_score(preds, gts, scale, self.oks_sigma)
            oks_sum += metrics["OKS"]

            self.results.append({"Video": image_path.name, **metrics})
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            n_images += 1

        self.reset()
        self.tp, self.fp, self.fn = totals["tp"], totals["fp"], totals["fn"]
        total_metrics = self.compute()
        if n_images:
            for thr in self.pck_thresholds:
                total_metrics[f"PCK@{thr:g}"] = pck_sums[thr] / n_images
            total_metrics["OKS"] = oks_sum / n_images
        else:
            for thr in self.pck_thresholds:
                total_metrics[f"PCK@{thr:g}"] = 0.0
            total_metrics["OKS"] = 0.0
        self.results.append({"Video": "TOTAL", **total_metrics})
        self.print_result()
        return self.results

    def print_result(self) -> None:
        pck_cols = [f"PCK@{thr:g}" for thr in self.pck_thresholds]
        columns = [
            "Video",
            "Total",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "OKS",
            *pck_cols,
        ]
        table = pd.DataFrame(self.results)
        present = [c for c in columns if c in table.columns]
        logger.info("\n" + table[present].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def compute(self) -> dict:
        eps = 1e-15
        total = self.tp + self.fp + self.fn
        # No TN for per-keypoint court matching (invisible GT ignored).
        accuracy = self.tp / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {
            "Total": total,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
