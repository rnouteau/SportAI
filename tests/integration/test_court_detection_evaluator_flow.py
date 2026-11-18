"""Integration: CourtDetectionEvaluator on synthetic image + mocked pose predict."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.evaluation import court_detection_evaluation as court_eval
from tests.evaluation.court_detection_evaluation import (
    N_KEYPOINTS,
    CourtDetectionEvaluator,
)


def _write_court_sample(directory: Path, stem: str = "court") -> tuple[Path, list[list[float]]]:
    """Write a PNG + YOLO-pose label; return image path and absolute GT keypoints."""
    width, height = 200, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 120, 40)
    image_path = directory / f"{stem}.png"
    assert cv2.imwrite(str(image_path), image)

    # Bbox centred 0.8×0.6 of the frame
    xc, yc, bw, bh = 0.5, 0.5, 0.8, 0.6
    parts = ["0", str(xc), str(yc), str(bw), str(bh)]
    gt_xy: list[list[float]] = []
    for i in range(N_KEYPOINTS):
        x_n = 0.1 + (i % 7) * 0.1
        y_n = 0.2 + (i // 7) * 0.3
        parts.extend([f"{x_n:.4f}", f"{y_n:.4f}", "2"])
        gt_xy.append([x_n * width, y_n * height])

    (directory / f"{stem}.txt").write_text(" ".join(parts), encoding="utf-8")
    return image_path, gt_xy


@pytest.mark.integration
def test_court_evaluator_perfect_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, gt_xy = _write_court_sample(tmp_path)

    monkeypatch.setattr(
        court_eval,
        "predict_keypoints",
        lambda model, image: [list(pt) for pt in gt_xy],
    )

    evaluator = CourtDetectionEvaluator(
        {
            "models": [{"name": "unused", "config": {}}],
            "evaluation": {
                "pck_thresholds": [0.05, 0.1],
                "match_threshold": 0.05,
                "oks_sigma": 0.05,
            },
        },
        directory=tmp_path,
    )
    results = evaluator.run_test()
    by_name = {row["Video"]: row for row in results}

    assert "court.png" in by_name
    perfect = by_name["court.png"]
    assert perfect["TP"] == N_KEYPOINTS
    assert perfect["FP"] == 0
    assert perfect["FN"] == 0
    assert perfect["F1-Score"] == pytest.approx(1.0)
    assert perfect["PCK@0.05"] == pytest.approx(1.0)
    assert perfect["OKS"] == pytest.approx(1.0)

    total = by_name["TOTAL"]
    assert total["F1-Score"] == pytest.approx(1.0)


@pytest.mark.integration
def test_court_evaluator_all_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_court_sample(tmp_path)

    monkeypatch.setattr(
        court_eval,
        "predict_keypoints",
        lambda model, image: [[999.0, 999.0] for _ in range(N_KEYPOINTS)],
    )

    evaluator = CourtDetectionEvaluator(
        {
            "models": [{"name": "unused", "config": {}}],
            "evaluation": {"match_threshold": 0.05, "pck_thresholds": [0.05]},
        },
        directory=tmp_path,
    )
    results = evaluator.run_test()
    row = next(r for r in results if r["Video"] == "court.png")
    assert row["TP"] == 0
    assert row["FN"] == N_KEYPOINTS
    assert row["FP"] == N_KEYPOINTS
    assert row["F1-Score"] == pytest.approx(0.0)
    assert row["PCK@0.05"] == pytest.approx(0.0)
