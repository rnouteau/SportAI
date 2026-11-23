"""PlayerEvaluator.run_test end to end: synthetic clips, real MOT files, fake tracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evaluation.player_evaluation import PlayerEvaluator
from tests.helpers import make_synthetic_video, pipeline_config, write_mot_clip

N_FRAMES = 6
NEAR = [40.0, 140.0, 80.0, 230.0]  # annotated id 1 (bottom of the image)
FAR = [200.0, 20.0, 230.0, 80.0]  # annotated id 2 (top of the image)
Track = list[float | int | None]


def _track(box: list[float], track_id: int | None) -> Track:
    return [*box, track_id]


def _both_players(n: int = N_FRAMES) -> dict[int, list[tuple[list[float], int]]]:
    return {f: [(NEAR, 1), (FAR, 2)] for f in range(n)}


def _evaluate(root: Path, tracks: list[list[Track]]) -> dict[str, dict]:
    config = pipeline_config(
        [("tests.fakes.FakeTrackingModel", {"tracks": tracks})], evaluation={"iou_threshold": 0.5}
    )
    results = PlayerEvaluator(config, root).run_test()
    json.dumps(results)  # the harness writes them to report.json
    return {row["Video"]: row for row in results}


def test_perfect_tracking_with_other_ids(tmp_path: Path) -> None:
    write_mot_clip(tmp_path, "clip", _both_players(), N_FRAMES)
    rows = _evaluate(tmp_path, [[_track(NEAR, 7), _track(FAR, 3)]] * N_FRAMES)

    assert set(rows) == {"clip.mp4", "TOTAL"}
    clip = rows["clip.mp4"]
    assert (clip["IDF1"], clip["MOTA"], clip["F1-Score"], clip["Mean IoU"]) == (1.0, 1.0, 1.0, 1.0)
    assert (clip["ID switches"], clip["FP"], clip["FN"], clip["GT"]) == (0, 0, 0, 2 * N_FRAMES)
    assert clip["Role accuracy"] == 0.0  # no player_id without the selector


def test_ids_swapped_mid_clip(tmp_path: Path) -> None:
    write_mot_clip(tmp_path, "clip", _both_players(), N_FRAMES)
    half = N_FRAMES // 2
    tracks = [[_track(NEAR, 7), _track(FAR, 3)]] * half
    tracks += [[_track(NEAR, 3), _track(FAR, 7)]] * (N_FRAMES - half)

    clip = _evaluate(tmp_path, tracks)["clip.mp4"]

    assert clip["ID switches"] == 2  # both players changed id once
    assert clip["IDF1"] < 1.0
    assert (clip["FP"], clip["FN"]) == (0, 0)


def test_close_up_misses_and_box_without_id(tmp_path: Path) -> None:
    """Frames 0-3 annotated, 4-5 are a close-up (no annotation).

    - frame 1: far player missed            -> 1 FN
    - frame 2: far player without track_id  -> found by the detection, missed by MOT
    - frame 5: a detection on the close-up  -> 1 FP
    """
    write_mot_clip(tmp_path, "clip", _both_players(4), N_FRAMES)
    tracks: list[list[Track]] = [
        [_track(NEAR, 7), _track(FAR, 3)],
        [_track(NEAR, 7)],
        [_track(NEAR, 7), _track(FAR, None)],
        [_track(NEAR, 7), _track(FAR, 3)],
        [],
        [_track([100.0, 50.0, 220.0, 240.0], 9)],
    ]

    clip = _evaluate(tmp_path, tracks)["clip.mp4"]

    assert (clip["GT"], clip["TP"], clip["FN"], clip["FP"]) == (8, 7, 1, 1)
    assert clip["Recall"] == pytest.approx(7 / 8)
    assert clip["Precision"] == pytest.approx(7 / 8)
    assert clip["MOTA"] == pytest.approx(1 - (2 + 1) / 8)  # MOT: 2 misses (incl. no id) + 1 FP


def test_total_row_and_video_without_annotation(tmp_path: Path) -> None:
    write_mot_clip(tmp_path, "clip_a", _both_players(), N_FRAMES)
    write_mot_clip(tmp_path, "clip_b", {f: [(NEAR, 1)] for f in range(N_FRAMES)}, N_FRAMES)
    make_synthetic_video(tmp_path / "no_gt.mp4", n_frames=N_FRAMES, size=(320, 240))

    rows = _evaluate(tmp_path, [[_track(NEAR, 7), _track(FAR, 3)]] * N_FRAMES)

    assert set(rows) == {"clip_a.mp4", "clip_b.mp4", "TOTAL"}  # no_gt.mp4 skipped
    assert rows["clip_b.mp4"]["FP"] == N_FRAMES  # the far player is not annotated in clip_b
    assert rows["TOTAL"]["GT"] == rows["clip_a.mp4"]["GT"] + rows["clip_b.mp4"]["GT"]
    assert rows["TOTAL"]["FP"] == N_FRAMES
