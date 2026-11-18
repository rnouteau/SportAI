"""Unit tests for court keypoint metrics (no model / no images)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.evaluation.court_detection_evaluation import (
    N_KEYPOINTS,
    bbox_scale,
    discover_image_label_pairs,
    keypoint_distance,
    match_keypoints_by_id,
    oks_score,
    parse_yolo_pose_label,
    pck_score,
)


@pytest.mark.unit
def test_bbox_scale() -> None:
    assert bbox_scale(4.0, 9.0) == 6.0


@pytest.mark.unit
def test_keypoint_distance() -> None:
    assert keypoint_distance([3.0, 4.0], (0.0, 0.0)) == 5.0
    assert keypoint_distance(None, (0.0, 0.0)) == float("inf")


@pytest.mark.unit
def test_pck_perfect() -> None:
    gts = [(10.0, 10.0, 2), (20.0, 20.0, 2)]
    preds = [[10.0, 10.0], [20.0, 20.0]]
    assert pck_score(preds, gts, scale=100.0, threshold=0.05) == 1.0


@pytest.mark.unit
def test_pck_ignores_invisible_gt() -> None:
    gts = [(10.0, 10.0, 0), (20.0, 20.0, 2)]
    preds = [[999.0, 999.0], [20.0, 20.0]]
    assert pck_score(preds, gts, scale=100.0, threshold=0.05) == 1.0


@pytest.mark.unit
def test_pck_miss() -> None:
    gts = [(0.0, 0.0, 2)]
    preds = [[100.0, 0.0]]
    # thr_px = 0.05 * 10 = 0.5 → miss
    assert pck_score(preds, gts, scale=10.0, threshold=0.05) == 0.0


@pytest.mark.unit
def test_oks_perfect() -> None:
    gts = [(1.0, 2.0, 2), (3.0, 4.0, 2)]
    preds = [[1.0, 2.0], [3.0, 4.0]]
    assert oks_score(preds, gts, scale=50.0, sigma=0.05) == 1.0


@pytest.mark.unit
def test_oks_far_is_low() -> None:
    gts = [(0.0, 0.0, 2)]
    preds = [[1000.0, 0.0]]
    score = oks_score(preds, gts, scale=10.0, sigma=0.05)
    assert 0.0 <= score < 0.01


@pytest.mark.unit
def test_match_keypoints_by_id_perfect() -> None:
    gts = [(0.0, 0.0, 2), (10.0, 0.0, 2)]
    preds = [[0.0, 0.0], [10.0, 0.0]]
    assert match_keypoints_by_id(preds, gts, scale=100.0, threshold=0.05) == (2, 0, 0)


@pytest.mark.unit
def test_match_keypoints_by_id_miss_and_fp_on_invisible() -> None:
    gts = [(0.0, 0.0, 2), (10.0, 0.0, 0)]
    preds: list = [None, [10.0, 0.0]]
    # visible miss → FN; invisible GT with pred → FP
    assert match_keypoints_by_id(preds, gts, scale=100.0, threshold=0.05) == (0, 1, 1)


@pytest.mark.unit
def test_match_keypoints_far_counts_fp_and_fn() -> None:
    gts = [(0.0, 0.0, 2)]
    preds = [[100.0, 0.0]]
    assert match_keypoints_by_id(preds, gts, scale=10.0, threshold=0.05) == (0, 1, 1)


def _yolo_pose_line(
    width: int,
    height: int,
    *,
    n_keypoints: int = N_KEYPOINTS,
    visible: bool = True,
) -> str:
    """One YOLO-pose line with bbox centred and keypoints on a grid (normalized)."""
    xc, yc, bw, bh = 0.5, 0.5, 0.8, 0.6
    parts = ["0", str(xc), str(yc), str(bw), str(bh)]
    v = 2 if visible else 0
    for i in range(n_keypoints):
        x_n = 0.1 + (i % 7) * 0.1
        y_n = 0.2 + (i // 7) * 0.2
        parts.extend([f"{x_n:.4f}", f"{y_n:.4f}", str(v)])
    return " ".join(parts)


@pytest.mark.unit
def test_parse_yolo_pose_label_happy_path(tmp_path: Path) -> None:
    width, height = 200, 100
    label_path = tmp_path / "court.txt"
    label_path.write_text(_yolo_pose_line(width, height), encoding="utf-8")

    bbox, keypoints = parse_yolo_pose_label(label_path, width, height)
    box_x, box_y, box_w, box_h = bbox
    assert box_x == pytest.approx(100.0)
    assert box_y == pytest.approx(50.0)
    assert box_w == pytest.approx(160.0)
    assert box_h == pytest.approx(60.0)
    assert len(keypoints) == N_KEYPOINTS
    assert keypoints[0][2] == 2
    assert keypoints[0][0] == pytest.approx(0.1 * width)
    assert keypoints[0][1] == pytest.approx(0.2 * height)


@pytest.mark.unit
def test_parse_yolo_pose_label_too_short(tmp_path: Path) -> None:
    label_path = tmp_path / "short.txt"
    label_path.write_text("0 0.5 0.5 0.8 0.6 0.1 0.2 2", encoding="utf-8")
    with pytest.raises(ValueError, match="too short"):
        parse_yolo_pose_label(label_path, 100, 100)


@pytest.mark.unit
def test_discover_image_label_pairs_flat(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "a.txt").write_text("0", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"fake")
    # orphan image without label
    (tmp_path / "c.jpg").write_bytes(b"fake")

    pairs = discover_image_label_pairs(tmp_path)
    names = {(img.name, lbl.name) for img, lbl in pairs}
    assert names == {("a.jpg", "a.txt")}


@pytest.mark.unit
def test_discover_image_label_pairs_images_labels_dirs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "court1.jpg").write_bytes(b"fake")
    (labels / "court1.txt").write_text("0", encoding="utf-8")
    (images / "orphan.jpg").write_bytes(b"fake")

    pairs = discover_image_label_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0].name == "court1.jpg"
    assert pairs[0][1].name == "court1.txt"
