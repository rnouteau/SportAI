"""Test doubles shared by the unit and integration tests (no weights, no real video).

Fake models are loaded by ``run_pipeline`` from their dotted name, e.g.
``"tests.fakes.FakeBallModel"``, and read what to write from their ``config``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from pipeline.models.model import Model
from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player import TennisPlayer
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class FakeVideoReader:
    """Stands for ``VideoReader``: black frames, ``info()`` and ``iter_frames()``."""

    def __init__(self, n_frames: int = 3, size: tuple[int, int] = (64, 48), fps: float = 30.0):
        self.frame_count = n_frames
        self.width, self.height = size
        self.fps = fps

    def info(self) -> dict:
        return {
            "video_path": "fake.mp4",
            "frame_count": self.frame_count,
            "fps": self.fps,
            "height": self.height,
            "width": self.width,
        }

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i in range(self.frame_count):
            yield i, np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeBallModel(Model[TennisAnalysis]):
    """Writes a ball at ``config["position"]`` (default ``[100, 100]``) on the frames whose id
    is a multiple of ``config["every"]`` (default 1 = every frame)."""

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        position = [float(v) for v in self.config.get("position", [100.0, 100.0])]
        every = int(self.config.get("every", 1))
        for frame in analysis.frames:
            frame.ball = TennisBall(position, confidence=0.9) if frame.id % every == 0 else None


class FakeTrackingModel(Model[TennisAnalysis]):
    """Writes ``config["tracks"][frame id]`` as the players of each frame.

    One entry per frame: ``[[x0, y0, x1, y1, track_id or None], ...]``. With
    ``config["matrix"]`` (image → court), every frame also gets that homography.
    """

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        tracks = self.config["tracks"]
        matrix = self.config.get("matrix")
        for frame in analysis.frames:
            if matrix is not None:
                frame.homography = TennisHomography(matrix=matrix)
            frame.players = [
                TennisPlayerAtFrame(
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    confidence=0.9,
                    track_id=None if track_id is None else int(track_id),
                )
                for x0, y0, x1, y1, track_id in tracks[frame.id]
            ]


class FakeSceneModel(Model[TennisAnalysis]):
    """Ball + one player (with court position) + identity homography on every frame."""

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            frame.ball = TennisBall(position=[20.0, 15.0], confidence=0.95)
            frame.players = [
                TennisPlayerAtFrame(
                    bbox=[5.0, 5.0, 25.0, 40.0],
                    confidence=0.8,
                    track_id=3,
                    court=CourtPosition(x=1.0, y=2.0),
                )
            ]
            frame.homography = TennisHomography(
                keypoints=[[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]],
                matrix=IDENTITY,
            )


class BallCountStat(Stat[TennisAnalysis]):
    """Stores the number of frames with a ball in ``final_stats`` (as a player name)."""

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        detected = sum(1 for frame in analysis.frames if frame.ball)
        analysis.final_stats.players = [TennisPlayer(id=1, name=f"{detected} balls")]
