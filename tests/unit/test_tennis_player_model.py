"""Unit tests for TennisPlayerModel (tracking output, missing ids, config defaults)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pipeline.models.tennis.tennis_player_model as tpm
from pipeline.models.tennis.tennis_player_model import TennisPlayerModel
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis


class _Boxes(SimpleNamespace):
    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeYolo:
    """Returns the prepared boxes and records the kwargs given to track()."""

    def __init__(self, per_frame: list[_Boxes]) -> None:
        self.per_frame = per_frame
        self.calls: list[dict] = []

    def track(self, image: np.ndarray, **kwargs: object):  # noqa: ANN201
        self.calls.append(kwargs)
        yield SimpleNamespace(boxes=self.per_frame[len(self.calls) - 1])


class _Reader:
    def __init__(self, n: int) -> None:
        self.n = n

    def iter_frames(self):  # noqa: ANN201
        for i in range(self.n):
            yield i, np.zeros((10, 10, 3), dtype=np.uint8)


def _boxes(confs: list[float], ids: list[int] | None) -> _Boxes:
    n = len(confs)
    return _Boxes(
        xyxy=torch.tensor([[float(i), 0.0, float(i) + 10, 20.0] for i in range(n)]),
        conf=torch.tensor(confs),
        id=torch.tensor(ids, dtype=torch.float32) if ids is not None else None,
    )


def _run(per_frame: list[_Boxes], config: dict | None = None, monkeypatch=None) -> tuple:  # noqa: ANN001
    monkeypatch.setattr(tpm, "Results", SimpleNamespace)
    model = TennisPlayerModel(config or {})
    fake = _FakeYolo(per_frame)
    model.model = fake  # type: ignore[assignment]
    frames = [{"id": i, "players": [], "ball": None} for i in range(len(per_frame))]
    analysis = TennisAnalysis({"sport": "tennis", "video": {}, "frames": frames})
    model.run(_Reader(len(per_frame)), analysis)  # type: ignore[arg-type]
    return analysis, fake


@pytest.mark.unit
def test_defaults_for_tracker_and_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _run([_boxes([0.9], [1])], monkeypatch=monkeypatch)
    assert fake.calls[0]["tracker"] == "bytetrack.yaml"
    assert fake.calls[0]["conf"] == pytest.approx(0.1)
    assert fake.calls[0]["persist"] is True


@pytest.mark.unit
def test_keeps_two_most_confident_with_track_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis, _ = _run([_boxes([0.3, 0.9, 0.8], [5, 7, 9])], monkeypatch=monkeypatch)
    players = analysis.frames[0].players
    assert [p.track_id for p in players] == [7, 9]
    assert [p.confidence for p in players] == pytest.approx([0.9, 0.8])


@pytest.mark.unit
def test_missing_ids_do_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ultralytics returns boxes without id when no track could be started."""
    analysis, _ = _run([_boxes([0.24], None), _boxes([0.9], [1])], monkeypatch=monkeypatch)
    first, second = analysis.frames[0].players, analysis.frames[1].players
    assert len(first) == 1 and first[0].track_id is None
    assert "track_id" not in first[0].data
    assert second[0].track_id == 1
