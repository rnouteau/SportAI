"""Unit tests for KeypointSmoother (temporal median of court keypoints)."""

from __future__ import annotations

import pytest

from pipeline.objects.keypoint_smoother import KeypointSmoother


def _pts(x: float, y: float, n: int = 3) -> list[list[float]]:
    return [[x + 10.0 * i, y] for i in range(n)]


@pytest.mark.unit
def test_window_one_is_passthrough() -> None:
    smoother = KeypointSmoother(window=1)
    pts, confs = smoother.update(_pts(100, 50), [0.9, 0.1, 0.8])
    assert pts == _pts(100, 50)
    assert confs == [0.9, 0.1, 0.8]


@pytest.mark.unit
def test_median_removes_single_frame_spike() -> None:
    smoother = KeypointSmoother(window=5, min_conf=0.3)
    for _ in range(4):
        smoother.update(_pts(100, 50), [0.9] * 3)
    pts, _ = smoother.update(_pts(130, 50), [0.9] * 3)  # one jittery frame
    assert pts[0] == pytest.approx([100.0, 50.0])


@pytest.mark.unit
def test_low_confidence_frames_are_ignored() -> None:
    smoother = KeypointSmoother(window=5, min_conf=0.5)
    smoother.update(_pts(100, 50), [0.9] * 3)
    pts, confs = smoother.update(_pts(400, 400), [0.1] * 3)
    assert pts[0] == pytest.approx([100.0, 50.0])
    assert confs[0] == pytest.approx(0.9)


@pytest.mark.unit
def test_point_never_confident_keeps_raw_value() -> None:
    smoother = KeypointSmoother(window=3, min_conf=0.5)
    pts, confs = smoother.update(_pts(100, 50), [0.9, 0.1, 0.9])
    assert pts[1] == pytest.approx([110.0, 50.0])
    assert confs[1] == pytest.approx(0.1)


@pytest.mark.unit
def test_scene_cut_resets_buffer() -> None:
    smoother = KeypointSmoother(window=5, min_conf=0.3, reset_px=20.0)
    for _ in range(4):
        smoother.update(_pts(100, 50), [0.9] * 3)
    pts, _ = smoother.update(_pts(600, 300), [0.9] * 3)  # new camera shot
    assert pts[0] == pytest.approx([600.0, 300.0])


@pytest.mark.unit
def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        KeypointSmoother(window=3).update(_pts(0, 0), [0.9])
