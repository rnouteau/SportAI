"""Player evaluator: detection, identity and role of the players, from one MOT annotation.

For every ``*.mp4``, the annotation is ``<video dir>/gt/gt.txt`` in MOT format (written
by the player annotator or CVAT)::

    frame (1-based), id, x, y, w, h, not_ignored, class_id, visibility

The annotated ids follow the player annotator convention: 1 = the near player (bottom
of the image), 2 = the far player.

One pipeline run gives three groups of metrics:

- **Detection** (boxes only, greedy IoU matching per frame): Precision, Recall,
  F1-Score, Mean IoU (how well the matched boxes fit) and the TP / FP / FN counts.
- **Identity** (motmetrics): IDF1 (share of the time each player keeps one id), ID
  switches, MOTA. The predicted id is ``player_id`` when the player selector ran,
  else ``track_id``; ids do not need to equal the annotated ones, only to be stable.
- **Role**: Role accuracy = share of the matched boxes whose ``player_id`` equals the
  annotated id, i.e. P1 really is the near player. IDF1 cannot see a P1 / P2 inversion.

Note: ``mm.distances.iou_matrix`` is not used because motmetrics 1.4.0 calls
``np.asfarray``, removed in NumPy 2; the matrix is built here with ``box_iou``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import motmetrics as mm
import numpy as np
import pandas as pd
from loguru import logger

from tests.evaluation.base import SampleResult, VideoEvaluator
from tests.evaluation.metrics import Box, Counts, box_iou, greedy_match, match_counts, safe_div

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "not_ignored", "class_id", "visibility"]
DEFAULT_IOU_THRESHOLD = 0.5

# motmetrics name -> name in the results / baseline
MOT_METRICS = {"idf1": "IDF1", "num_switches": "ID switches", "mota": "MOTA"}


class GroundTruthBox(NamedTuple):
    box: list[float]
    id: int


class PredictedBox(NamedTuple):
    box: list[float]
    track_id: int | None
    player_id: int | None

    @property
    def id(self) -> int | None:
        """Id used for the identity metrics: the selector's ``player_id`` first."""
        return self.player_id if self.player_id is not None else self.track_id


GroundTruth = dict[int, list[GroundTruthBox]]  # frame (0-based) -> boxes
Predictions = dict[int, list[PredictedBox]]


def load_mot_ground_truth(path: Path) -> GroundTruth:
    """MOT ``gt.txt`` → ``{frame (0-based): [GroundTruthBox(xyxy, id), ...]}``.

    Rows with ``not_ignored == 0`` are dropped.
    """
    df = pd.read_csv(path, header=None, names=MOT_COLUMNS)
    df = df[df["not_ignored"] != 0]
    frames: GroundTruth = defaultdict(list)
    for row in df.itertuples(index=False):
        x, y, w, h = float(row.x), float(row.y), float(row.w), float(row.h)
        frames[int(row.frame) - 1].append(GroundTruthBox([x, y, x + w, y + h], int(row.id)))
    return dict(frames)


def predictions_by_frame(frames: list[dict]) -> Predictions:
    """Pipeline frames (``analysis["frames"]``) → ``{frame: [PredictedBox, ...]}``."""
    preds: Predictions = {}
    for frame in frames:
        boxes = []
        for player in frame.get("players") or []:
            bbox = player.get("bbox") if player else None
            if not bbox:
                continue
            boxes.append(
                PredictedBox(
                    [float(v) for v in bbox[:4]], player.get("track_id"), player.get("player_id")
                )
            )
        preds[int(frame["id"])] = boxes
    return preds


def distance_matrix(gt_boxes: list[Box], pred_boxes: list[Box], iou_threshold: float) -> np.ndarray:
    """motmetrics distance matrix: ``1 - IoU``, NaN (= cannot match) when IoU < threshold."""
    matrix = np.full((len(gt_boxes), len(pred_boxes)), np.nan)
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou >= iou_threshold:
                matrix[i, j] = 1.0 - iou
    return matrix


@dataclass
class PlayerScores:
    """Everything summed over frames; TOTAL is the sum of the per-video scores."""

    counts: Counts = field(default_factory=Counts)
    iou_sum: float = 0.0
    role_checked: int = 0  # matched boxes with a player_id
    role_correct: int = 0
    accumulators: list[mm.MOTAccumulator] = field(default_factory=list)

    def __add__(self, other: PlayerScores) -> PlayerScores:
        return PlayerScores(
            self.counts + other.counts,
            self.iou_sum + other.iou_sum,
            self.role_checked + other.role_checked,
            self.role_correct + other.role_correct,
            self.accumulators + other.accumulators,
        )

    @property
    def mean_iou(self) -> float:
        return safe_div(self.iou_sum, self.counts.tp)

    @property
    def role_accuracy(self) -> float:
        return safe_div(self.role_correct, self.role_checked)


def score_detection_and_role(
    gts: GroundTruth, preds: Predictions, iou_threshold: float
) -> PlayerScores:
    scores = PlayerScores()
    for frame in sorted(set(gts) | set(preds)):
        frame_gts, frame_preds = gts.get(frame, []), preds.get(frame, [])
        matches = greedy_match(
            [p.box for p in frame_preds], [g.box for g in frame_gts], iou_threshold
        )
        scores.counts += match_counts(len(frame_preds), len(frame_gts), len(matches))
        for pi, gi, iou in matches:
            scores.iou_sum += iou
            player_id = frame_preds[pi].player_id
            if player_id is not None:
                scores.role_checked += 1
                scores.role_correct += int(player_id == frame_gts[gi].id)
    return scores


def accumulate(gts: GroundTruth, preds: Predictions, iou_threshold: float) -> mm.MOTAccumulator:
    """Feed every frame to a MOT accumulator; predicted boxes without any id are left out."""
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gts) | set(preds)):
        frame_gts = gts.get(frame, [])
        frame_preds = [p for p in preds.get(frame, []) if p.id is not None]
        acc.update(
            [g.id for g in frame_gts],
            [p.id for p in frame_preds],
            distance_matrix(
                [g.box for g in frame_gts], [p.box for p in frame_preds], iou_threshold
            ),
        )
    return acc


def identity_metrics(accumulators: list[mm.MOTAccumulator]) -> dict[str, float | int]:
    """IDF1 / ID switches / MOTA of one accumulator, or of several merged (TOTAL)."""
    metrics = mm.metrics.create()
    if len(accumulators) == 1:
        summary = metrics.compute(accumulators[0], metrics=list(MOT_METRICS), name="all")
        row = summary.loc["all"]
    else:
        names = [str(i) for i in range(len(accumulators))]
        summary = metrics.compute_many(
            accumulators, metrics=list(MOT_METRICS), names=names, generate_overall=True
        )
        row = summary.loc["OVERALL"]
    return {
        "IDF1": float(row["idf1"]),
        "ID switches": int(row["num_switches"]),
        "MOTA": float(row["mota"]),
    }


def player_row(scores: PlayerScores) -> dict:
    counts = scores.counts
    return {
        "GT": counts.tp + counts.fn,
        "TP": counts.tp,
        "FP": counts.fp,
        "FN": counts.fn,
        **counts.scores(),
        "Mean IoU": scores.mean_iou,
        **identity_metrics(scores.accumulators),
        "Role accuracy": scores.role_accuracy,
    }


class PlayerEvaluator(VideoEvaluator):
    columns = (
        "Precision",
        "Recall",
        "F1-Score",
        "Mean IoU",
        "IDF1",
        "ID switches",
        "MOTA",
        "Role accuracy",
    )

    def __init__(self, config: dict, directory: Path) -> None:
        super().__init__(config, directory)
        self.iou_threshold = float(self.evaluation.get("iou_threshold", DEFAULT_IOU_THRESHOLD))

    def ground_truth_path(self, video_path: Path) -> Path:
        return video_path.parent / "gt" / "gt.txt"

    def evaluate_frames(self, frames: list[dict], gt_path: Path) -> SampleResult:
        gts = load_mot_ground_truth(gt_path)
        preds = predictions_by_frame(frames)
        without_id = sum(p.id is None for boxes in preds.values() for p in boxes)
        if without_id:
            logger.warning(f"{gt_path.parent.parent.name}: {without_id} boxes without any id")

        scores = score_detection_and_role(gts, preds, self.iou_threshold)
        scores.accumulators.append(accumulate(gts, preds, self.iou_threshold))
        if scores.counts.tp and not scores.role_checked:
            logger.warning("No player_id on the matched boxes: Role accuracy is 0 (selector off?)")
        return SampleResult(player_row(scores), scores)

    def total(self, results: list[SampleResult]) -> dict:
        return player_row(sum((r.extra for r in results), PlayerScores()))
