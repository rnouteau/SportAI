from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import distance

from pipeline.run import run_pipeline


class BallTrackingEvaluator:
    def __init__(self, config: dict, directory: Path) -> None:
        logger.info("Initialized BallTrackingEvaluator")
        self.config = config
        self.directory = directory
        self.min_dist = config.get("evaluation", {}).get("min_dist", 10)
        self.reset()

    def run_test(self) -> list[dict]:
        self.results = []
        totals = {"tp": 0, "fp1": 0, "fp2": 0, "tn": 0, "fn": 0}

        for video_path in sorted(Path(self.directory).rglob("*.mp4")):
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                logger.warning(f"No annotations for {video_path.name}, skipped")
                continue

            result = run_pipeline("tennis", str(video_path), self.config, test=True)
            gt_df = pd.read_csv(csv_path)

            self.reset()
            self.match_prediction(result["frames"], gt_df)
            self.results.append({"Video": video_path.name, **self.compute()})

            for key in totals:
                totals[key] += getattr(self, key)

        self.reset()
        for key, value in totals.items():
            setattr(self, key, value)
        self.results.append({"Video": "TOTAL", **self.compute()})

        self.print_result()
        return self.results

    def reset(self) -> None:
        self.tp, self.fp1, self.fp2, self.fp, self.tn, self.fn = 0, 0, 0, 0, 0, 0

    def print_result(self) -> None:
        columns = ["Video", "Total", "Accuracy", "Precision", "Recall", "F1-Score"]
        table = pd.DataFrame(self.results)[columns]
        logger.info("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def match_prediction(self, frames: list[dict], gt_df: pd.DataFrame) -> None:
        pred_df = pd.DataFrame(
            [
                {
                    "frame": f["id"],
                    "x_pred": f["ball"]["position"][0] if f["ball"] else np.nan,
                    "y_pred": f["ball"]["position"][1] if f["ball"] else np.nan,
                    "confidence": f["ball"]["confidence"] if f["ball"] else np.nan,
                }
                for f in frames
            ]
        )

        matched = gt_df.merge(pred_df, on="frame", how="left")

        for _, row in matched.iterrows():
            x, y = row["x"], row["y"]
            x_pred, y_pred = row["x_pred"], row["y_pred"]

            if not pd.isna(x_pred):
                if not pd.isna(x):
                    dist = distance.euclidean((x, y), (x_pred, y_pred))
                    if dist < self.min_dist:
                        self.tp += 1
                    else:
                        self.fp1 += 1
                else:
                    self.fp2 += 1
            else:
                if not pd.isna(x):
                    self.fn += 1
                else:
                    self.tn += 1

    def compute(self) -> dict:
        eps = 1e-15
        self.fp = self.fp1 + self.fp2
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {
            "Total": total,
            "TP": self.tp,
            "FP1": self.fp1,
            "FP2": self.fp2,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
