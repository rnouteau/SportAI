"""Unit tests for TennisCourtModel preprocessing (stretch → rescale) and run()."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from pipeline.models.tennis.tennis_court_model import (
    TennisCourtModel,
    parse_input_size,
    predict_court_keypoints,
)
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_court_schema import N_POINTS, TennisCourtSchema

FRAME_W, FRAME_H = 1920, 1080


class _Keypoints(SimpleNamespace):
    def __len__(self) -> int:
        return int(self.xy.shape[0])


class _FakeYolo:
    """Returns fixed keypoints expressed in the coordinates of the image it receives."""

    def __init__(self, kpts_in_input: np.ndarray, conf: float = 0.9) -> None:
        self.kpts = kpts_in_input
        self.conf = conf
        self.calls: list[tuple[tuple[int, ...], dict]] = []

    def predict(self, source: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append((source.shape, kwargs))
        xy = torch.tensor(self.kpts, dtype=torch.float32).unsqueeze(0)
        conf = torch.full((1, self.kpts.shape[0]), self.conf)
        return [SimpleNamespace(keypoints=_Keypoints(xy=xy, conf=conf))]


@pytest.mark.unit
def test_parse_input_size() -> None:
    assert parse_input_size(640) == (640, 640)
    assert parse_input_size([360, 640]) == (360, 640)
    assert parse_input_size(None) == (640, 640)
    with pytest.raises(ValueError):
        parse_input_size("big")


@pytest.mark.unit
def test_stretch_resizes_to_square_and_rescales_keypoints() -> None:
    fake = _FakeYolo(np.array([[320.0, 320.0], [640.0, 0.0]]))
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    out = predict_court_keypoints(fake, image, input_size=(640, 640), stretch=True)  # type: ignore[arg-type]

    assert out is not None
    pts, confs = out
    assert fake.calls[0][0][:2] == (640, 640)
    assert pts[0] == pytest.approx([960.0, 540.0])
    assert pts[1] == pytest.approx([1920.0, 0.0])
    assert confs == pytest.approx([0.9, 0.9])


@pytest.mark.unit
def test_no_stretch_passes_frame_unchanged() -> None:
    fake = _FakeYolo(np.array([[100.0, 200.0]]))
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    out = predict_court_keypoints(fake, image, stretch=False)  # type: ignore[arg-type]

    assert out is not None
    assert fake.calls[0][0][:2] == (FRAME_H, FRAME_W)
    assert out[0][0] == pytest.approx([100.0, 200.0])


@pytest.mark.unit
def test_run_writes_homography_in_frame_pixels() -> None:
    """Keypoints predicted in the stretched 640x640 space must give the true court."""
    schema = TennisCourtSchema()
    court = np.array([schema.model_xy(i) for i in range(N_POINTS)], dtype=np.float64)
    h_c2i = cv2.getPerspectiveTransform(
        np.float32([[-5.485, 11.885], [5.485, 11.885], [-5.485, -11.885], [5.485, -11.885]]),
        np.float32([[650, 315], [1250, 315], [410, 770], [1495, 775]]),
    ).astype(np.float64)
    frame_pts = cv2.perspectiveTransform(court.reshape(-1, 1, 2), h_c2i).reshape(-1, 2)
    stretched_pts = frame_pts * np.array([640.0 / FRAME_W, 640.0 / FRAME_H])

    model = TennisCourtModel({"smoothing_window": 3})
    model.model = _FakeYolo(stretched_pts)  # type: ignore[assignment]

    class _Reader:
        def iter_frames(self):  # noqa: ANN202
            for i in range(3):
                yield i, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    frames = [{"id": i, "players": [], "ball": None} for i in range(3)]
    analysis = TennisAnalysis({"sport": "tennis", "video": {}, "frames": frames})
    model.run(_Reader(), analysis)  # type: ignore[arg-type]

    for frame in analysis.frames:
        homo = frame.homography
        assert homo is not None and homo.matrix is not None
        projected = schema.image_to_court(np.asarray(homo.matrix), frame_pts)
        assert np.allclose(projected, court, atol=0.05)
