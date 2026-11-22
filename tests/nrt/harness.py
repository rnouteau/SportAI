"""Shared helpers for NRT regression tests (prepare config, context, report, assert)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.assets import resolve_weights
from tests.baseline import build_baseline, compare_to_baseline, load_baseline, save_baseline
from utils import read_yaml
from utils.paths import get_root


def prepare_config(path: Path) -> dict:
    config = read_yaml(path)
    device = os.environ.get("SPORTAI_DEVICE")
    for model_cfg in config["models"]:
        if device:
            model_cfg["config"]["device"] = device
        # A model may name its own Hugging Face weights (several models in one test).
        weights_cfg = model_cfg.get("weights") or config["weights"]
        model_cfg["config"]["weight"] = resolve_weights(model_cfg["config"], weights_cfg)
    return config


def device_for_context(model_cfg: dict) -> str:
    device = str(model_cfg.get("device", "cpu"))
    if device.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def base_context(config: dict) -> dict:
    model_cfg = config["models"][0]["config"]
    return {
        "device": device_for_context(model_cfg),
        "dataset_revision": config["dataset"].get("revision", "local"),
        "weights": Path(model_cfg["weight"]).name,
    }


def save_report(
    results: list[dict],
    baseline: dict | None,
    current: dict,
    reg: dict,
    out_dir: Path,
    *,
    title: str,
    count_columns: list[str],
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

    present_counts = [c for c in count_columns if c in pd.DataFrame(results).columns]
    counts = pd.DataFrame(results)[present_counts]
    comparison = pd.DataFrame(rows)

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        f.write(
            f"Context: `{json.dumps(current['context'], ensure_ascii=False)}` - "
            f"tolerance: {reg['tolerance']}\n\n"
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


def assert_vs_baseline(
    results: list[dict],
    config: dict,
    context: dict,
    report_dir: Path,
    *,
    title: str,
    count_columns: list[str],
    update_baseline: bool,
) -> None:
    reg = config["regression"]
    current = build_baseline(results, context, reg["metrics"])
    baseline_path = get_root() / reg["baseline"]

    if update_baseline or not baseline_path.exists():
        save_baseline(baseline_path, current)
        save_report(
            results,
            None,
            current,
            reg,
            report_dir,
            title=title,
            count_columns=count_columns,
        )
        pytest.skip(f"Baseline written to {baseline_path}: review it, then commit it")

    baseline = load_baseline(baseline_path)
    save_report(
        results,
        baseline,
        current,
        reg,
        report_dir,
        title=title,
        count_columns=count_columns,
    )
    problems = compare_to_baseline(current, baseline, reg["tolerance"])
    assert not problems, "Results differ from the baseline:\n" + "\n".join(problems)
