"""Unit tests for shared court YOLO-pose encode/decode."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.court_annotator.court_yolo import (
    decode_pose_label_line,
    encode_pose_label_line,
    load_pose_label_points,
    write_pose_label,
)

from pipeline.objects.tennis.tennis_court_schema import N_POINTS
from tests.evaluation.court_detection_evaluation import N_KEYPOINTS, parse_yolo_pose_label


@pytest.mark.unit
def test_n_points_matches_eval_alias() -> None:
    assert N_POINTS == N_KEYPOINTS == 14


@pytest.mark.unit
def test_encode_decode_round_trip() -> None:
    width, height = 200, 100
    points = {
        0: (20.0, 10.0),
        3: (180.0, 10.0),
        10: (20.0, 90.0),
        13: (180.0, 90.0),
    }
    line = encode_pose_label_line(points, width, height)
    recovered = decode_pose_label_line(line, width, height)
    assert set(recovered) == set(points)
    for pid, (x, y) in points.items():
        assert recovered[pid][0] == pytest.approx(x, abs=1e-3)
        assert recovered[pid][1] == pytest.approx(y, abs=1e-3)


@pytest.mark.unit
def test_encode_invisible_slots() -> None:
    width, height = 100, 100
    points = {0: (10.0, 10.0)}
    line = encode_pose_label_line(points, width, height)
    parts = line.split()
    assert len(parts) == 5 + N_POINTS * 3
    # keypoint 1 (missing) → v=0
    assert parts[5 + 3 + 2] == "0"


@pytest.mark.unit
def test_write_and_parse_compatible_with_evaluator(tmp_path: Path) -> None:
    width, height = 160, 120
    points = {1: (40.0, 30.0), 5: (80.0, 40.0), 8: (80.0, 90.0)}
    label_path = tmp_path / "court.txt"
    write_pose_label(label_path, points, width, height)

    loaded = load_pose_label_points(label_path, width, height)
    assert set(loaded) == set(points)

    _bbox, kpts = parse_yolo_pose_label(label_path, width, height)
    assert len(kpts) == N_POINTS
    assert kpts[1][2] == 2
    assert kpts[1][0] == pytest.approx(40.0, abs=1e-3)
    assert kpts[0][2] == 0
