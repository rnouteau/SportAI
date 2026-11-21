"""Integration: PlayerTrackingEvaluator.run_test end to end with a fake tracking model.

Real synthetic videos and real MOT ``gt.txt`` files go through ``run_pipeline``; only
the YOLO + ByteTrack model is replaced by a fake that writes the boxes and ids given
in its config (one entry per frame: ``[[x0, y0, x1, y1, track_id or None], ...]``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.video_reader import VideoReader
from tests.evaluation.player_tracking_evaluation import PlayerTrackingEvaluator
from tests.helpers import make_synthetic_video

N_FRAMES = 6
SIZE = (320, 240)
P1 = [40.0, 140.0, 80.0, 230.0]  # near player (bottom of the image)
P2 = [200.0, 20.0, 230.0, 80.0]  # far player (top of the image)
MODEL_NAME = "tests.integration.test_player_tracking_evaluator_flow.FakeTrackingModel"


class FakeTrackingModel(Model[TennisAnalysis]):
    """Writes ``config["tracks"][frame]`` as the players of each frame."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.tracks: list[list[list[float | int | None]]] = config["tracks"]

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            assert frame.id is not None
            players = []
            for x0, y0, x1, y1, track_id in self.tracks[frame.id]:
                player = TennisPlayerAtFrame.create(position=[x0, y0, x1, y1], confidence=0.9)
                player.track_id = None if track_id is None else int(track_id)
                players.append(player)
            frame.players = players


def _row(box: list[float], track_id: int | None) -> list[float | int | None]:
    return [*box, track_id]


def _write_clip(root: Path, name: str, gt: dict[int, list[tuple[list[float], int]]]) -> Path:
    """``<root>/<name>/<name>.mp4`` + ``<root>/<name>/gt/gt.txt`` (MOT, frames 1-based)."""
    clip_dir = root / name
    (clip_dir / "gt").mkdir(parents=True)
    make_synthetic_video(clip_dir / f"{name}.mp4", n_frames=N_FRAMES, size=SIZE)
    lines = []
    for frame in sorted(gt):
        for (x0, y0, x1, y1), gt_id in gt[frame]:
            lines.append(
                f"{frame + 1},{gt_id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},1,1,1.0"
            )
    (clip_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return clip_dir


def _config(tracks: list[list[list[float | int | None]]]) -> dict:
    return {
        "analysis": "pipeline.objects.tennis.TennisAnalysis",
        "models": [{"name": MODEL_NAME, "config": {"tracks": tracks}}],
        "evaluation": {"iou_threshold": 0.5},
    }


def _both_players(n: int = N_FRAMES) -> dict[int, list[tuple[list[float], int]]]:
    return {f: [(P1, 1), (P2, 2)] for f in range(n)}


def _by_video(results: list[dict]) -> dict[str, dict]:
    return {row["Video"]: row for row in results}


@pytest.mark.integration
def test_perfect_tracking_with_other_ids(tmp_path: Path) -> None:
    """Predicted ids 7 / 3 instead of 1 / 2: only consistency matters."""
    _write_clip(tmp_path, "clip", _both_players())
    tracks = [[_row(P1, 7), _row(P2, 3)] for _ in range(N_FRAMES)]

    results = PlayerTrackingEvaluator(_config(tracks), tmp_path).run_test()

    rows = _by_video(results)
    assert set(rows) == {"clip.mp4", "TOTAL"}
    clip = rows["clip.mp4"]
    assert clip["IDF1"] == pytest.approx(1.0)
    assert clip["MOTA"] == pytest.approx(1.0)
    assert (clip["ID switches"], clip["FP"], clip["FN"], clip["GT"]) == (0, 0, 0, 2 * N_FRAMES)
    json.dumps(results)  # harness.py writes the results to report.json


@pytest.mark.integration
def test_ids_swapped_mid_clip(tmp_path: Path) -> None:
    _write_clip(tmp_path, "clip", _both_players())
    half = N_FRAMES // 2
    tracks = [[_row(P1, 7), _row(P2, 3)] for _ in range(half)]
    tracks += [[_row(P1, 3), _row(P2, 7)] for _ in range(N_FRAMES - half)]

    clip = _by_video(PlayerTrackingEvaluator(_config(tracks), tmp_path).run_test())["clip.mp4"]

    assert clip["ID switches"] == 2  # both players changed id once
    assert clip["IDF1"] < 1.0
    assert (clip["FP"], clip["FN"]) == (0, 0)


@pytest.mark.integration
def test_close_up_misses_and_missing_ids(tmp_path: Path) -> None:
    """Frames 0-3 annotated, 4-5 are a close-up (no ground truth).

    - frame 1: P2 missed by the model                     -> 1 FN
    - frame 2: P2 predicted without track_id (ignored)    -> 1 FN
    - frame 5: a detection on the close-up                -> 1 FP
    """
    _write_clip(tmp_path, "clip", _both_players(4))
    tracks: list[list[list[float | int | None]]] = [
        [_row(P1, 7), _row(P2, 3)],
        [_row(P1, 7)],
        [_row(P1, 7), _row(P2, None)],
        [_row(P1, 7), _row(P2, 3)],
        [],
        [_row([100.0, 50.0, 220.0, 240.0], 9)],
    ]

    clip = _by_video(PlayerTrackingEvaluator(_config(tracks), tmp_path).run_test())["clip.mp4"]

    assert clip["GT"] == 8
    assert (clip["FN"], clip["FP"], clip["ID switches"]) == (2, 1, 0)
    assert clip["Recall"] == pytest.approx(6 / 8)
    assert clip["Precision"] == pytest.approx(6 / 7)


@pytest.mark.integration
def test_total_row_and_video_without_ground_truth(tmp_path: Path) -> None:
    _write_clip(tmp_path, "clip_a", _both_players())
    _write_clip(tmp_path, "clip_b", {f: [(P1, 1)] for f in range(N_FRAMES)})
    make_synthetic_video(tmp_path / "no_gt.mp4", n_frames=N_FRAMES, size=SIZE)
    tracks = [[_row(P1, 7), _row(P2, 3)] for _ in range(N_FRAMES)]

    rows = _by_video(PlayerTrackingEvaluator(_config(tracks), tmp_path).run_test())

    assert set(rows) == {"clip_a.mp4", "clip_b.mp4", "TOTAL"}  # no_gt.mp4 skipped
    assert rows["clip_b.mp4"]["FP"] == N_FRAMES  # P2 predicted but not in clip_b's ground truth
    assert rows["TOTAL"]["GT"] == rows["clip_a.mp4"]["GT"] + rows["clip_b.mp4"]["GT"]
    assert rows["TOTAL"]["FP"] == N_FRAMES
