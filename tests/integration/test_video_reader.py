from pathlib import Path

import numpy as np
import pytest

from pipeline.video_reader import VideoReader
from tests.helpers import make_synthetic_video


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


def test_release_closes_capture(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=2)

    reader = VideoReader(str(video_path))

    _ = reader.read_frame(0)
    reader.release()


def test_iter_frames_length_and_order(tmp_path: Path) -> None:
    n_frames = 4
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames, size=(32, 24))
    sequential = VideoReader(str(video_path))
    random_access = VideoReader(str(video_path))
    try:
        frames = list(sequential.iter_frames())
        assert len(frames) == n_frames
        assert [idx for idx, _ in frames] == list(range(n_frames))
        for idx, image in frames:
            assert np.array_equal(image, random_access.read_frame(idx))
    finally:
        sequential.release()
        random_access.release()


def test_iter_triplets_length_and_edges(tmp_path: Path) -> None:
    n_frames = 5
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames)
    reader = VideoReader(str(video_path))
    try:
        triplets = list(reader.iter_triplets())
        assert len(triplets) == n_frames

        index0, prev0, curr0, next0 = triplets[0]
        assert index0 == 0
        assert np.array_equal(prev0, curr0)
        assert not np.array_equal(curr0, next0)

        index_last, prev_last, curr_last, next_last = triplets[-1]
        assert index_last == n_frames - 1
        assert np.array_equal(curr_last, next_last)
        assert not np.array_equal(prev_last, curr_last)
    finally:
        reader.release()


def test_iter_triplets_matches_read_frame(tmp_path: Path) -> None:
    n_frames = 6
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=n_frames, size=(32, 24))
    sequential = VideoReader(str(video_path))
    random_access = VideoReader(str(video_path))
    try:
        triplets = list(sequential.iter_triplets())
        for index, prev, curr, nxt in triplets:
            prev_idx = max(index - 1, 0)
            next_idx = min(index + 1, n_frames - 1)
            assert np.array_equal(prev, random_access.read_frame(prev_idx))
            assert np.array_equal(curr, random_access.read_frame(index))
            assert np.array_equal(nxt, random_access.read_frame(next_idx))
    finally:
        sequential.release()
        random_access.release()


def test_iter_triplets_single_frame(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=1)
    reader = VideoReader(str(video_path))
    try:
        triplets = list(reader.iter_triplets())
        assert len(triplets) == 1
        index, prev, curr, nxt = triplets[0]
        assert index == 0
        assert np.array_equal(prev, curr)
        assert np.array_equal(curr, nxt)
    finally:
        reader.release()
