"""Metric helpers shared by every evaluator: counts → scores, box IoU, box matching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Box = Sequence[float]  # x0, y0, x1, y1 (pixels)


def safe_div(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, 0.0 when the denominator is 0."""
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class Counts:
    """Confusion counts. ``tn`` only makes sense when a sample can be "nothing to find"."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return safe_div(2 * self.precision * self.recall, self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        return safe_div(self.tp + self.tn, self.total)

    def scores(self) -> dict[str, float]:
        """Precision / Recall / F1-Score, the names used in reports and baselines."""
        return {"Precision": self.precision, "Recall": self.recall, "F1-Score": self.f1}


def box_iou(a: Box, b: Box) -> float:
    """IoU of two xyxy boxes (corners may be given in any order)."""
    ax0, ax1 = sorted((float(a[0]), float(a[2])))
    ay0, ay1 = sorted((float(a[1]), float(a[3])))
    bx0, bx1 = sorted((float(b[0]), float(b[2])))
    by0, by1 = sorted((float(b[1]), float(b[3])))

    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return safe_div(inter, union)


def greedy_match(
    preds: Sequence[Box], gts: Sequence[Box], iou_threshold: float
) -> list[tuple[int, int, float]]:
    """One-to-one matching by decreasing IoU: ``[(pred index, gt index, iou), ...]``.

    Only pairs with ``iou >= iou_threshold`` are matched. Unmatched predictions are
    false positives, unmatched ground-truth boxes are misses.
    """
    pairs = sorted(
        ((box_iou(pred, gt), pi, gi) for pi, pred in enumerate(preds) for gi, gt in enumerate(gts)),
        reverse=True,
    )
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pi, gi in pairs:
        if iou < iou_threshold:
            break
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matches.append((pi, gi, iou))
    return matches


def match_counts(n_preds: int, n_gts: int, n_matches: int) -> Counts:
    """Counts of one frame once its boxes are matched."""
    return Counts(tp=n_matches, fp=n_preds - n_matches, fn=n_gts - n_matches)
