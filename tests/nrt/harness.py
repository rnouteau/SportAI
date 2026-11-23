"""Run one NRT suite: prepare its config, evaluate, write the report, compare to the baseline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from tests.assets import resolve_dataset_dir, resolve_weights
from tests.baseline import build_baseline, compare_to_baseline, load_baseline, save_baseline
from tests.nrt.suites import ROOT, Suite
from utils import read_yaml

# Model config keys that are not settings of the run (they are reported elsewhere).
_NOT_SETTINGS = {"device", "weight"}


def prepare_config(path: Path) -> dict:
    """Read a suite config, apply ``SPORTAI_DEVICE`` and resolve every model's weights."""
    config = read_yaml(path)
    device = os.environ.get("SPORTAI_DEVICE")
    for model_cfg in config["models"]:
        if device:
            model_cfg["config"]["device"] = device
        # A model may name its own Hugging Face weights (several models in one suite).
        weights_cfg = model_cfg.get("weights") or config["weights"]
        model_cfg["config"]["weight"] = resolve_weights(model_cfg["config"], weights_cfg)
    return config


def effective_device(requested: str) -> str:
    """Device the run really uses: CUDA falls back to CPU when it is not available."""
    if requested.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def short_name(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def run_context(config: dict) -> dict:
    """Everything a baseline depends on, flat so that it reads well in the report and wiki.

    A change of context (weights, settings, dataset...) stops the comparison and asks for
    a new baseline instead of reporting metric differences.
    """
    first_model = config["models"][0]["config"]
    context: dict = {
        "device": effective_device(str(first_model.get("device", "cpu"))),
        "dataset_revision": config["dataset"].get("revision", "local"),
    }
    for model in config["models"]:
        name = short_name(model["name"])
        context[f"{name}.weights"] = Path(model["config"]["weight"]).name
        for key, value in sorted(model["config"].items()):
            if key not in _NOT_SETTINGS:
                context[f"{name}.{key}"] = value
    stats = [short_name(stat["name"]) for stat in config.get("stats") or []]
    if stats:
        context["stats"] = ", ".join(stats)
    for key, value in sorted((config.get("evaluation") or {}).items()):
        context[f"evaluation.{key}"] = value
    return context


def comparison_rows(current: dict, baseline: dict | None, tolerance: float) -> list[dict]:
    rows = []
    expected_videos = (baseline or {}).get("videos", {})
    for video, metrics in current["videos"].items():
        expected_metrics = expected_videos.get(video, {})
        for metric, value in metrics.items():
            expected = expected_metrics.get(metric)
            delta = None if expected is None else value - expected
            if baseline is None:
                status = "new baseline"
            elif delta is None:
                status = "missing from baseline"
            else:
                status = "OK" if abs(delta) <= tolerance else "DIFF"
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
    return rows


def save_report(
    suite: Suite, results: list[dict], current: dict, baseline: dict | None, tolerance: float
) -> None:
    """``report.md`` (read by people) and ``report.json`` (read by the wiki renderer)."""
    out_dir = suite.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    counts = table[["Video", *(c for c in suite.count_columns if c in table.columns)]]
    comparison = pd.DataFrame(comparison_rows(current, baseline, tolerance))

    context = json.dumps(current["context"], ensure_ascii=False)
    lines = [
        f"## {suite.title} - non-regression",
        "",
        f"Context: `{context}` - tolerance: {tolerance}",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Counts",
        "",
        counts.to_markdown(index=False),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"current": current, "baseline": baseline, "results": results}
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@dataclass
class SuiteOutcome:
    results: list[dict]
    baseline_written: bool = False
    problems: list[str] = field(default_factory=list)


def run_suite(suite: Suite, *, update_baseline: bool = False) -> SuiteOutcome:
    """Evaluate a suite, write its report, then write or check its baseline."""
    config = prepare_config(suite.config_path)
    evaluator = suite.load_evaluator()(config, resolve_dataset_dir(config["dataset"]))
    results = evaluator.run_test()
    n_samples = sum(row["Video"] != "TOTAL" for row in results)
    if n_samples < suite.min_samples:
        raise AssertionError(f"{suite.id}: {n_samples} annotated sample(s) found, dataset missing?")

    regression = config["regression"]
    tolerance = float(regression["tolerance"])
    current = build_baseline(results, run_context(config), regression["metrics"])
    baseline_path = ROOT / regression["baseline"]

    if update_baseline or not baseline_path.exists():
        save_baseline(baseline_path, current)
        save_report(suite, results, current, None, tolerance)
        return SuiteOutcome(results, baseline_written=True)

    baseline = load_baseline(baseline_path)
    save_report(suite, results, current, baseline, tolerance)
    return SuiteOutcome(results, problems=compare_to_baseline(current, baseline, tolerance))
