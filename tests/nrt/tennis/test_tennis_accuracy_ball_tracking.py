import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.assets import resolve_dataset_dir, resolve_weights
from tests.baseline import build_baseline, compare_to_baseline, load_baseline, save_baseline
from tests.evaluation.ball_tracking import BallTrackingEvaluator
from utils import read_yaml
from utils.paths import get_root

config_file = (
    get_root()
    / "tests"
    / "configs"
    / "nrt"
    / "tennis"
    / "config_test_tennis_accuracy_ball_tracking.yaml"
)
report_dir = get_root() / "outputs" / "tests" / "tennis_ball_tracking"


@pytest.mark.nrt
def test_tennis_ball_tracking_regression(update_baseline: bool) -> None:
    config = prepare_config(config_file)

    results = BallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
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


def save_report(
    results: list[dict], baseline: dict | None, current: dict, reg: dict, out_dir: Path
) -> None:
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
            rows.append(
                {
                    "Video": video,
                    "Metric": metric,
                    "Current": value,
                    "Baseline": expected,
                    "Delta": delta,
                    "Status": status,
                }
            )

    counts = pd.DataFrame(results)[["Video", "Total", "TP", "FP", "TN", "FN"]]
    comparison = pd.DataFrame(rows)

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("## Tennis ball tracking - non-regression\n\n")
        f.write(
            f"Context: `{json.dumps(current['context'], ensure_ascii=False)}` - tolerance: {reg['tolerance']}\n\n"
        )
        f.write(comparison.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### Counts\n\n")
        f.write(counts.to_markdown(index=False))
        f.write("\n")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"current": current, "baseline": baseline, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    config = prepare_config(config_file)
    BallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
