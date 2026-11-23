"""NRT harness and suite registry (no model, no dataset)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.baseline import load_baseline
from tests.nrt import harness
from tests.nrt.suites import ROOT, SUITES, SUITES_BY_ID
from utils import read_yaml


def test_prepare_config_resolves_each_model_weights(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    config = {
        "models": [
            {"name": "player", "config": {"weight": "./models/player.pt"}},
            {
                "name": "court",
                "config": {"weight": "./models/court.pt"},
                "weights": {"repo_id": "r", "filename": "court/court.pt"},
            },
        ],
        "weights": {"repo_id": "r", "filename": "player/player.pt"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(
        harness, "resolve_weights", lambda model, weights: calls.append(weights["filename"]) or "w"
    )
    monkeypatch.setenv("SPORTAI_DEVICE", "cpu")

    prepared = harness.prepare_config(path)

    assert calls == ["player/player.pt", "court/court.pt"]
    assert [m["config"]["device"] for m in prepared["models"]] == ["cpu", "cpu"]


def test_run_context_lists_every_model_stat_and_setting() -> None:
    config = {
        "models": [
            {"name": "a.TennisPlayerModel", "config": {"device": "cpu", "weight": "/x/yolo.pt"}},
            {
                "name": "a.TennisCourtModel",
                "config": {"device": "cpu", "weight": "/y/court.pt", "min_conf": 0.3},
            },
        ],
        "stats": [{"name": "a.TennisSideStat"}],
        "dataset": {"revision": "v4"},
        "evaluation": {"iou_threshold": 0.5},
    }
    assert harness.run_context(config) == {
        "device": "cpu",
        "dataset_revision": "v4",
        "TennisPlayerModel.weights": "yolo.pt",
        "TennisCourtModel.weights": "court.pt",
        "TennisCourtModel.min_conf": 0.3,
        "stats": "TennisSideStat",
        "evaluation.iou_threshold": 0.5,
    }


def test_cuda_falls_back_to_cpu_in_the_context(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(harness.torch.cuda, "is_available", lambda: False)
    assert harness.effective_device("cuda:0") == "cpu"


def test_comparison_rows_status() -> None:
    current = {"videos": {"a": {"F1": 0.9, "New": 1.0}}}
    baseline = {"videos": {"a": {"F1": 0.8}}}
    rows = harness.comparison_rows(current, baseline, tolerance=0.01)
    assert [(r["Metric"], r["Status"]) for r in rows] == [
        ("F1", "DIFF"),
        ("New", "missing from baseline"),
    ]
    assert harness.comparison_rows(current, None, 0.01)[0]["Status"] == "new baseline"


@pytest.mark.parametrize("suite", SUITES, ids=[s.id for s in SUITES])
def test_every_suite_is_consistent(suite) -> None:  # noqa: ANN001
    """Config, evaluator, baseline and report columns of a suite agree with each other."""
    config = read_yaml(suite.config_path)
    regression = config["regression"]
    assert Path(regression["baseline"]).stem == suite.id
    assert suite.main_metric in regression["metrics"]
    evaluator = suite.load_evaluator()
    assert hasattr(evaluator, "run_test")

    baseline_path = ROOT / regression["baseline"]
    if baseline_path.exists():
        baseline = load_baseline(baseline_path)
        assert baseline["videos"], "empty baseline"
        for metrics in baseline["videos"].values():
            assert set(metrics) == set(regression["metrics"])


def test_suite_ids_are_unique() -> None:
    assert len(SUITES_BY_ID) == len(SUITES)
