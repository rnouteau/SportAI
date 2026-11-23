"""Ball evaluator: one predicted ball per frame vs the annotated ball (``<video>.csv``).

Annotation: ``frame, x, y`` with empty ``x``/``y`` when the ball is not visible.
Per frame, with ``d`` the distance between prediction and annotation:

- TP  : ball annotated, ball predicted, ``d < min_dist``
- FP1 : ball annotated, ball predicted too far (``d >= min_dist``)
- FP2 : no ball annotated, ball predicted
- FN  : ball annotated, nothing predicted
- TN  : no ball annotated, nothing predicted

There is at most one ball per frame, so Accuracy (TP + TN over all frames) is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tests.evaluation.base import SampleResult, VideoEvaluator
from tests.evaluation.metrics import Counts

DEFAULT_MIN_DIST = 10.0


@dataclass
class BallCounts:
    tp: int = 0
    fp1: int = 0  # predicted too far from the annotated ball
    fp2: int = 0  # predicted while no ball is annotated
    fn: int = 0
    tn: int = 0

    def __add__(self, other: BallCounts) -> BallCounts:
        return BallCounts(
            self.tp + other.tp,
            self.fp1 + other.fp1,
            self.fp2 + other.fp2,
            self.fn + other.fn,
            self.tn + other.tn,
        )

    def counts(self) -> Counts:
        return Counts(tp=self.tp, fp=self.fp1 + self.fp2, fn=self.fn, tn=self.tn)

    def row(self) -> dict:
        counts = self.counts()
        return {
            "Total": counts.total,
            "TP": self.tp,
            "FP1": self.fp1,
            "FP2": self.fp2,
            "FP": counts.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Accuracy": counts.accuracy,
            **counts.scores(),
        }


def predicted_balls(frames: list[dict]) -> dict[int, tuple[float, float]]:
    """``{frame id: (x, y)}`` for the frames with a predicted ball."""
    return {
        int(frame["id"]): (float(frame["ball"]["position"][0]), float(frame["ball"]["position"][1]))
        for frame in frames
        if frame.get("ball")
    }


def match_balls(frames: list[dict], gt_df: pd.DataFrame, min_dist: float) -> BallCounts:
    """Compare the predictions with every annotated frame of ``gt_df``."""
    preds = predicted_balls(frames)
    counts = BallCounts()
    for row in gt_df.itertuples(index=False):
        annotated = not (pd.isna(row.x) or pd.isna(row.y))
        pred = preds.get(int(row.frame))
        if pred is None:
            if annotated:
                counts.fn += 1
            else:
                counts.tn += 1
        elif not annotated:
            counts.fp2 += 1
        elif math.dist(pred, (float(row.x), float(row.y))) < min_dist:
            counts.tp += 1
        else:
            counts.fp1 += 1
    return counts


class BallEvaluator(VideoEvaluator):
    columns = ("Total", "Accuracy", "Precision", "Recall", "F1-Score")

    def __init__(self, config: dict, directory: Path) -> None:
        super().__init__(config, directory)
        self.min_dist = float(self.evaluation.get("min_dist", DEFAULT_MIN_DIST))

    def ground_truth_path(self, video_path: Path) -> Path:
        return video_path.with_suffix(".csv")

    def evaluate_frames(self, frames: list[dict], gt_path: Path) -> SampleResult:
        counts = match_balls(frames, pd.read_csv(gt_path), self.min_dist)
        return SampleResult(counts.row(), counts)

    def total(self, results: list[SampleResult]) -> dict:
        return sum((r.extra for r in results), BallCounts()).row()
