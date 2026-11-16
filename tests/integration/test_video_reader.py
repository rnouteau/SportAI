from pathlib import Path

import numpy as np
import pytest

from pipeline.video_reader import VideoReader
from tests.helpers import make_synthetic_video


@pytest.mark.integration
def test_read_synthetic_video_metadata(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4")

    video_reader = VideoReader(str(video_path))

    try:
        assert video_reader.frame_count == 5
        assert video_reader.fps == pytest.approx(10, 0.1)
        assert video_reader.height == 48
        assert video_reader.width == 64
        assert video_reader.video_path == str(video_path)
    finally:
        video_reader.release()


@pytest.mark.integration
def test_read_frame_returns_image(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3, size=(64, 48))

    reader = VideoReader(str(video_path))
    try:
        frame = reader.read_frame(0)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
    finally:
        reader.release()


@pytest.mark.integration
def test_read_frame_out_of_bounds_raises(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3)

    reader = VideoReader(str(video_path))
    try:
        with pytest.raises(ValueError, match="out of bounds"):
            reader.read_frame(-1)
        with pytest.raises(ValueError, match="out of bounds"):
            reader.read_frame(3)
    finally:
        reader.release()


@pytest.mark.integration
def test_release_closes_capture(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=2)

    reader = VideoReader(str(video_path))

    _ = reader.read_frame(0)
    reader.release()
