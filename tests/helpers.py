from pathlib import Path

import cv2
import numpy as np


def make_synthetic_video(
    path: Path,
    n_frames: int = 5,
    size: tuple[int, int] = (64, 48),
    fps: float = 10.0,
) -> Path:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 40) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
