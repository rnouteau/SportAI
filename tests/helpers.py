"""Small builders shared by the tests (synthetic videos, analyses, configs, annotations)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_court_schema import N_POINTS, model_xy
from pipeline.objects.tennis.tennis_frame import TennisFrame

TENNIS_ANALYSIS = "pipeline.objects.tennis.TennisAnalysis"
COURT_OFFSET = (100.0, 200.0)


def make_synthetic_video(
    path: Path,
    n_frames: int = 5,
    size: tuple[int, int] = (64, 48),
    fps: float = 10.0,
) -> Path:
    """Tiny ``.mp4`` whose frames are plain grey levels."""
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for i in range(n_frames):
        writer.write(np.full((height, width, 3), (i * 40) % 255, dtype=np.uint8))
    writer.release()
    return path


def make_tennis_analysis(n_frames: int = 3, fps: float = 30.0) -> TennisAnalysis:
    """Empty tennis analysis (no ball, no player, no homography) of ``n_frames`` frames."""
    return TennisAnalysis(
        sport="tennis",
        video={"video_path": "fake.mp4", "frame_count": n_frames, "fps": fps},
        frames=[TennisFrame(id=i) for i in range(n_frames)],
    )


def pipeline_config(
    models: list[tuple[str, dict]], stats: list[str] | None = None, **extra
) -> dict:
    """``run_pipeline`` config: ``models`` = ``[(dotted name, config), ...]``."""
    config: dict = {
        "analysis": TENNIS_ANALYSIS,
        "models": [{"name": name, "config": cfg} for name, cfg in models],
        **extra,
    }
    if stats:
        config["stats"] = [{"name": name} for name in stats]
    return config


def write_mot_clip(
    root: Path,
    name: str,
    gt: dict[int, list[tuple[list[float], int]]],
    n_frames: int,
    size: tuple[int, int] = (320, 240),
) -> Path:
    """``<root>/<name>/<name>.mp4`` + ``<root>/<name>/gt/gt.txt`` (MOT, frames 1-based).

    ``gt`` = ``{frame (0-based): [([x0, y0, x1, y1], id), ...]}``.
    """
    clip_dir = root / name
    (clip_dir / "gt").mkdir(parents=True)
    make_synthetic_video(clip_dir / f"{name}.mp4", n_frames=n_frames, size=size)
    lines = [
        f"{frame + 1},{gt_id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},1,1,1.0"
        for frame in sorted(gt)
        for (x0, y0, x1, y1), gt_id in gt[frame]
    ]
    (clip_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return clip_dir


def shifted_court_keypoints(offset: tuple[float, float] = COURT_OFFSET) -> list[list[float]]:
    """The 14 court points (metres) shifted by ``offset``: an image that is the court translated."""
    return [[model_xy(i)[0] + offset[0], model_xy(i)[1] + offset[1]] for i in range(N_POINTS)]
