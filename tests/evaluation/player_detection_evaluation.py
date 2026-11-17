"""Player detection evaluator: IoU matching with COCO-style mAP50 / mAP50-95."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipeline.run import run_pipeline

# COCO mAP50:95 IoU thresholds (TP requires at least 0.5)
COCO_IOU_THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(0.5, 1.0, 0.05))


def box_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """IoU of two xyxy boxes."""
    ax0, ay0 = float(min(a[0], a[2])), float(min(a[1], a[3]))
    ax1, ay1 = float(max(a[0], a[2])), float(max(a[1], a[3]))
    bx0, by0 = float(min(b[0], b[2])), float(min(b[1], b[3]))
    bx1, by1 = float(max(b[0], b[2])), float(max(b[1], b[3]))

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def greedy_match(
    preds: list[list[float]],
    gts: list[list[float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one matching by decreasing IoU.

    Returns (tp, fp, fn). A pair is TP only if IoU >= iou_threshold
    (COCO mAP50:95 minimum is 0.5); otherwise the prediction counts as FP.
    """
    if not preds and not gts:
        return 0, 0, 0
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0

    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            pairs.append((box_iou(pred, gt), pi, gi))
    pairs.sort(reverse=True, key=lambda item: item[0])

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    tp = 0
    for iou, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        if iou < iou_threshold:
            break
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1

    fp = len(preds) - len(matched_pred)
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def average_precision_at_iou(
    frame_preds: dict[int, list[list[float]]],
    frame_gts: dict[int, list[list[float]]],
    iou_threshold: float,
) -> float:
    """
    Single-class micro precision at a fixed IoU threshold (AP proxy when scores tie).

    Aggregates greedy TP/FP/FN over frames; returns TP/(TP+FP).
    """
    tp = fp = fn = 0
    for frame_idx in sorted(set(frame_preds) | set(frame_gts)):
        t, f, n = greedy_match(
            frame_preds.get(frame_idx, []),
            frame_gts.get(frame_idx, []),
            iou_threshold,
        )
        tp += t
        fp += f
        fn += n
    denom = tp + fp
    if denom == 0:
        return 1.0 if fn == 0 else 0.0
    return float(tp / denom)


class PlayerDetectionEvaluator:
    def __init__(self, config: dict, directory: Path) -> None:
        logger.info("Initialized PlayerDetectionEvaluator")
        self.config = config
        self.directory = directory
        self.iou_threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))
        self.reset()

    def run_test(self) -> list[dict]:
        self.results = []
        totals = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        map50_values: list[float] = []
        map5095_values: list[float] = []

        for video_path in sorted(Path(self.directory).rglob("*.mp4")):
            csv_path = video_path.with_name(f"{video_path.stem}_players.csv")
            if not csv_path.exists():
                logger.warning(f"No annotations for {video_path.name}, skipped")
                continue

            result = run_pipeline("tennis", str(video_path), self.config, test=True)
            gt_df = pd.read_csv(csv_path).dropna(subset=["x0", "y0", "x1", "y1"])

            self.reset()
            frame_preds, frame_gts = self._boxes_by_frame(result["frames"], gt_df)
            self.match_prediction(frame_preds, frame_gts)

            map50 = average_precision_at_iou(frame_preds, frame_gts, 0.5)
            map5095 = float(
                np.mean(
                    [
                        average_precision_at_iou(frame_preds, frame_gts, t)
                        for t in COCO_IOU_THRESHOLDS
                    ]
                )
            )
            metrics = self.compute()
            metrics["mAP50"] = map50
            metrics["mAP50-95"] = map5095
            self.results.append({"Video": video_path.name, **metrics})
            map50_values.append(map50)
            map5095_values.append(map5095)

            for key in totals:
                totals[key] += getattr(self, key)

        self.reset()
        for key, value in totals.items():
            setattr(self, key, value)
        total_metrics = self.compute()
        total_metrics["mAP50"] = float(np.mean(map50_values)) if map50_values else 0.0
        total_metrics["mAP50-95"] = float(np.mean(map5095_values)) if map5095_values else 0.0
        self.results.append({"Video": "TOTAL", **total_metrics})

        self.print_result()
        return self.results

    def reset(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0

    def print_result(self) -> None:
        columns = [
            "Video",
            "Total",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP50-95",
        ]
        table = pd.DataFrame(self.results)[columns]
        logger.info("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    @staticmethod
    def _boxes_by_frame(
        frames: list[dict],
        gt_df: pd.DataFrame,
    ) -> tuple[dict[int, list[list[float]]], dict[int, list[list[float]]]]:
        frame_gts: dict[int, list[list[float]]] = defaultdict(list)
        for row in gt_df.itertuples(index=False):
            frame_gts[int(row.frame)].append(
                [float(row.x0), float(row.y0), float(row.x1), float(row.y1)]
            )

        frame_preds: dict[int, list[list[float]]] = {}
        for frame in frames:
            frame_id = int(frame["id"])
            boxes: list[list[float]] = []
            for player in frame.get("players") or []:
                if not player or not player.get("position"):
                    continue
                pos = player["position"]
                boxes.append([float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3])])
            frame_preds[frame_id] = boxes
        return frame_preds, dict(frame_gts)

    def match_prediction(
        self,
        frame_preds: dict[int, list[list[float]]],
        frame_gts: dict[int, list[list[float]]],
    ) -> None:
        # Pipeline emits every frame id (players may be []); empty-empty → TN
        frames = sorted(set(frame_preds) | set(frame_gts))
        for frame_idx in frames:
            preds = frame_preds.get(frame_idx, [])
            gts = frame_gts.get(frame_idx, [])
            if not preds and not gts:
                self.tn += 1
                continue
            tp, fp, fn = greedy_match(preds, gts, self.iou_threshold)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def compute(self) -> dict:
        eps = 1e-15
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {
            "Total": total,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
