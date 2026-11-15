import copy
import pytest
from tests.baseline import (
    build_baseline,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]

def sample_context(**overrides) -> dict:
    context = {
        "device": "cpu",
        "dataset_revision": "v1",
        "weights": "TENNIS_BALL_MODEL_04_11_2026.pt",
        "threshold": 0.5,
        "min_dist": 10,
    }
    context.update(overrides)
    return context


def sample_baseline() -> dict:
    return {
        "context": sample_context(),
        "videos": {
            "clip_a.mp4": {
                "Accuracy": 0.8715,
                "Precision": 0.9926,
                "Recall": 0.7952,
                "F1-Score": 0.8830,
            },
            "TOTAL": {
                "Accuracy": 0.8715,
                "Precision": 0.9926,
                "Recall": 0.7952,
                "F1-Score": 0.8830,
            },
        },
    }

@pytest.mark.unit
def test_build_baseline_rounds_and_indexes_by_video() -> None:
    results = [
        {
            "Video": "clip_a.mp4",
            "Accuracy": 0.87154,
            "Precision": 0.99261,
            "Recall": 0.79519,
            "F1-Score": 0.88304,
        },
        {
            "Video": "TOTAL",
            "Accuracy": 0.87154,
            "Precision": 0.99261,
            "Recall": 0.79519,
            "F1-Score": 0.88304,
        },
    ]

    context = sample_context()
    baseline = build_baseline(results,context,METRICS)

    assert baseline["context"] == context
    assert set(baseline["videos"]) == {"clip_a.mp4","TOTAL"}
    assert baseline["videos"]["clip_a.mp4"]["Accuracy"] == 0.8715
    assert baseline["videos"]["clip_a.mp4"]["Precision"] == 0.9926
    assert baseline["videos"]["clip_a.mp4"]["Recall"] == 0.7952
    assert baseline["videos"]["clip_a.mp4"]["F1-Score"] == 0.8830
    assert baseline["videos"]["TOTAL"]["F1-Score"] == 0.8830

@pytest.mark.unit
def test_save_and_load_round_trip(tmp_path) -> None:
    baseline = sample_baseline()
    path = tmp_path / "nested"/ "tennis_ball_tracking.json"
    save_baseline(path,baseline)

    assert path.exists()

    loaded = load_baseline(path)

    assert loaded == baseline
    
@pytest.mark.unit
def test_compare_identical_returns_empty() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)

    assert compare_to_baseline(current, baseline, tolerance=0.005) == []

@pytest.mark.unit
def test_compare_small_delta_within_tolerance_is_ok() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)
    current["videos"]["clip_a.mp4"]["Recall"] += 0.003

    assert compare_to_baseline(current, baseline, tolerance=0.005) == []


@pytest.mark.unit
def test_compare_large_drop_and_rise_are_reported() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)
    current["videos"]["TOTAL"]["Recall"] -= 0.05
    current["videos"]["TOTAL"]["F1-Score"] += 0.02
    problems = compare_to_baseline(current, baseline, tolerance=0.005)

    assert "TOTAL Recall: 0.7952 -> 0.7452 (-0.0500)" in problems
    assert "TOTAL F1-Score: 0.8830 -> 0.9030 (+0.0200)" in problems

@pytest.mark.unit
def test_compare_missing_video_from_run() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)
    del current["videos"]["clip_a.mp4"]
    problems = compare_to_baseline(current, baseline, tolerance=0.005)

    assert "clip_a.mp4: video in the baseline but missing from the run" in problems

@pytest.mark.unit
def test_compare_extra_video_in_run() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)
    current["videos"]["clip_b.mp4"] = {
        "Accuracy": 0.9,
        "Precision": 0.9,
        "Recall": 0.9,
        "F1-Score": 0.9,
    }
    problems = compare_to_baseline(current, baseline, tolerance=0.005)
    assert "clip_b.mp4: video missing from the baseline" in problems


@pytest.mark.unit
def test_compare_context_change_stops_before_metric_diffs() -> None:
    baseline = sample_baseline()
    current = copy.deepcopy(baseline)
    current["context"]["threshold"] = 0.6
    current["videos"]["TOTAL"]["Recall"] = 0.0
    problems = compare_to_baseline(current, baseline, tolerance=0.005)

    assert any("context.threshold" in p for p in problems)
    assert (
        "The run context changed: regenerate the baseline with --update-baseline"
        in problems
    )
    assert not any("Recall:" in p for p in problems)