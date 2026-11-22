"""Player tracking evaluator: MOT metrics (IDF1, ID switches, MOTA...) with motmetrics.

For every ``*.mp4`` under the dataset directory, the ground truth is read from
``<video dir>/gt/gt.txt`` (MOT format, written by the player annotator or CVAT):

    frame (1-based), track_id, x, y, w, h, not_ignored, class_id, visibility

The pipeline predictions (``players[].position`` in xyxy + ``players[].track_id``)
are matched to the ground truth frame by frame with an IoU distance matrix.
Predicted ids do not need to equal the ground-truth ids: motmetrics only measures
whether each id stays on the same player.

Note: ``mm.distances.iou_matrix`` is not used because motmetrics 1.4.0 calls
``np.asfarray``, removed in NumPy 2; the matrix is built here with ``box_iou``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd
from loguru import logger

from pipeline.run import run_pipeline

Box = list[float]  # x0, y0, x1, y1
FrameBoxes = dict[int, list[tuple[Box, int]]]  # frame -> [(box, id), ...]

MOT_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "not_ignored", "class_id", "visibility"]

# motmetrics name -> name in the results / baseline
METRICS = {
    "idf1": "IDF1",
    "num_switches": "ID switches",
    "mota": "MOTA",
    "precision": "Precision",
    "recall": "Recall",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_objects": "GT",
}
COUNT_METRICS = {"ID switches", "FP", "FN", "GT"}


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


def load_mot_ground_truth(gt_path: Path) -> FrameBoxes:
    """Read a MOT ``gt.txt`` → {frame (0-based): [(xyxy box, id), ...]}.

    Rows marked ``not_ignored == 0`` are skipped.
    """
    df = pd.read_csv(gt_path, header=None)
    df = df.iloc[:, : len(MOT_COLUMNS)]
    df.columns = MOT_COLUMNS[: df.shape[1]]
    if "not_ignored" in df.columns:
        df = df[df["not_ignored"].fillna(1) != 0]

    frame_gts: FrameBoxes = defaultdict(list)
    for row in df.itertuples(index=False):
        x, y, w, h = float(row.x), float(row.y), float(row.w), float(row.h)
        frame_gts[int(row.frame) - 1].append(([x, y, x + w, y + h], int(row.track_id)))
    return dict(frame_gts)


def predictions_by_frame(frames: list[dict]) -> tuple[FrameBoxes, int]:
    """Pipeline frames → ({frame: [(xyxy box, id), ...]}, boxes skipped for missing id).

    The id is ``player_id`` when the player selector ran, otherwise ``track_id``.
    """
    frame_preds: FrameBoxes = defaultdict(list)
    without_id = 0
    for frame in frames:
        frame_id = int(frame["id"])
        for player in frame.get("players") or []:
            if not player or not player.get("position"):
                continue
            # player_id (stable, set by TennisPlayerSelectorStat) wins over the raw tracker id.
            pred_id = player.get("player_id", player.get("track_id"))
            if pred_id is None:
                without_id += 1
                continue
            pos = player["position"]
            box = [float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3])]
            frame_preds[frame_id].append((box, int(pred_id)))
    return dict(frame_preds), without_id


def distance_matrix(gt_boxes: list[Box], pred_boxes: list[Box], iou_threshold: float) -> np.ndarray:
    """Rows = ground truth, columns = predictions; ``1 - IoU``, NaN when IoU < threshold."""
    matrix = np.full((len(gt_boxes), len(pred_boxes)), np.nan)
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = box_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                matrix[i, j] = 1.0 - iou
    return matrix


def accumulate(
    frame_gts: FrameBoxes, frame_preds: FrameBoxes, iou_threshold: float
) -> mm.MOTAccumulator:
    """Feed every frame that has ground truth or predictions to a MOT accumulator."""
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(frame_gts) | set(frame_preds)):
        gts = frame_gts.get(frame, [])
        preds = frame_preds.get(frame, [])
        acc.update(
            [gt_id for _, gt_id in gts],
            [pred_id for _, pred_id in preds],
            distance_matrix([b for b, _ in gts], [b for b, _ in preds], iou_threshold),
        )
    return acc


def summary_row(summary: pd.DataFrame, name: str) -> dict:
    """One motmetrics summary row → plain-Python dict with readable metric names."""
    row = summary.loc[name]
    out: dict[str, float | int] = {}
    for mm_name, label in METRICS.items():
        value = row[mm_name]
        out[label] = int(value) if label in COUNT_METRICS else float(value)
    return out


def compute_metrics(acc: mm.MOTAccumulator) -> dict:
    summary = mm.metrics.create().compute(acc, metrics=list(METRICS), name="video")
    return summary_row(summary, "video")


class PlayerTrackingEvaluator:
    def __init__(self, config: dict, directory: Path) -> None:
        logger.info("Initialized PlayerTrackingEvaluator")
        self.config = config
        self.directory = directory
        self.iou_threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))
        self.results: list[dict] = []

    def run_test(self) -> list[dict]:
        self.results = []
        accumulators: list[mm.MOTAccumulator] = []
        names: list[str] = []

        for video_path in sorted(Path(self.directory).rglob("*.mp4")):
            gt_path = video_path.parent / "gt" / "gt.txt"
            if not gt_path.exists():
                logger.warning(f"No ground truth for {video_path.name}, skipped")
                continue

            result = run_pipeline("tennis", str(video_path), self.config, test=True)
            if result is None:
                raise RuntimeError(f"Pipeline returned nothing for {video_path.name}")

            frame_gts = load_mot_ground_truth(gt_path)
            frame_preds, without_id = predictions_by_frame(result["frames"])
            if without_id:
                logger.warning(
                    f"{video_path.name}: {without_id} predicted boxes without track_id ignored"
                )

            acc = accumulate(frame_gts, frame_preds, self.iou_threshold)
            accumulators.append(acc)
            names.append(video_path.name)
            self.results.append({"Video": video_path.name, **compute_metrics(acc)})

        if accumulators:
            overall = mm.metrics.create().compute_many(
                accumulators, metrics=list(METRICS), names=names, generate_overall=True
            )
            self.results.append({"Video": "TOTAL", **summary_row(overall, "OVERALL")})

        self.print_result()
        return self.results

    def print_result(self) -> None:
        if not self.results:
            logger.warning("No video evaluated")
            return
        columns = ["Video", *METRICS.values()]
        table = pd.DataFrame(self.results)[columns]
        logger.info("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
