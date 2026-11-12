
import json
import os
from pathlib import Path
from pipeline.run import run_pipeline
from scipy.spatial import distance
from tests.assets import resolve_dataset_dir, resolve_weights
from tests.baseline import build_baseline, compare_to_baseline, load_baseline, save_baseline
from utils import read_yaml
from loguru import logger
import pandas as pd
import numpy as np
import pytest
import torch

from utils.paths import get_root

config_file = get_root() / "tests" / "configs" / "nrt" / "tennis" / "config_test_tennis_accuracy_ball_tracking.yaml"
report_dir = get_root() / "outputs" / "tests" / "tennis_ball_tracking"


@pytest.mark.nrt
def test_tennis_ball_tracking_regression(update_baseline: bool) -> None:
    config = prepare_config(config_file)

    results = TennisBallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
    assert len(results) > 1, "no annotated video found"

    reg = config["regression"]
    current = build_baseline(results, run_context(config), reg["metrics"])
    baseline_path = get_root() / reg["baseline"]

    if update_baseline or not baseline_path.exists():
        save_baseline(baseline_path, current)
        save_report(results, None, current, reg, report_dir)
        pytest.skip(f"Baseline written to {baseline_path}: review it, then commit it")

    baseline = load_baseline(baseline_path)
    save_report(results, baseline, current, reg, report_dir)
    problems = compare_to_baseline(current, baseline, reg["tolerance"])
    assert not problems, "Results differ from the baseline:\n" + "\n".join(problems)


def prepare_config(path: Path) -> dict:
    config = read_yaml(path)
    device = os.environ.get("SPORTAI_DEVICE")
    for model_cfg in config["models"]:
        if device:
            model_cfg["config"]["device"] = device
        model_cfg["config"]["weight"] = resolve_weights(model_cfg["config"], config["weights"])
    return config


def run_context(config: dict) -> dict:
    model_cfg = config["models"][0]["config"]
    device = str(model_cfg.get("device", "cpu"))
    if device.startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "dataset_revision": config["dataset"]["revision"],
        "weights": Path(model_cfg["weight"]).name,
        "threshold": model_cfg.get("threshold", 0.5),
        "min_dist": config["evaluation"]["min_dist"],
    }


def save_report(results: list[dict], baseline: dict | None, current: dict, reg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video, metrics in current["videos"].items():
        expected_metrics = (baseline or {}).get("videos", {}).get(video, {})
        for metric, value in metrics.items():
            expected = expected_metrics.get(metric)
            delta = None if expected is None else value - expected
            if baseline is None:
                status = "new baseline"
            elif expected is None:
                status = "missing from baseline"
            else:
                status = "OK" if abs(delta) <= reg["tolerance"] else "DIFF"
            rows.append({"Video": video, "Metric": metric, "Current": value,
                         "Baseline": expected, "Delta": delta, "Status": status})

    counts = pd.DataFrame(results)[["Video", "Total", "TP", "FP", "TN", "FN"]]
    comparison = pd.DataFrame(rows)

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("## Tennis ball tracking - non-regression\n\n")
        f.write(f"Context: `{json.dumps(current['context'], ensure_ascii=False)}` - tolerance: {reg['tolerance']}\n\n")
        f.write(comparison.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### Counts\n\n")
        f.write(counts.to_markdown(index=False))
        f.write("\n")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"current": current, "baseline": baseline, "results": results}, f, indent=2, ensure_ascii=False)


class TennisBallTrackingEvaluator:
    def __init__(self, config: dict, directory: Path) -> None:
        logger.info("Initialized TennisBallTrackingEvaluator")
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
        pred_df = pd.DataFrame([
            {
                "frame": f["id"],
                "x_pred": f["ball"]["position"][0] if f["ball"] else np.nan,
                "y_pred": f["ball"]["position"][1] if f["ball"] else np.nan,
                "confidence": f["ball"]["confidence"] if f["ball"] else np.nan,
            }
            for f in frames
        ])

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


if __name__ == "__main__":
    config = prepare_config(config_file)
    TennisBallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
