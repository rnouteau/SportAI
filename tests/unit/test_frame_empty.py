"""Unit tests for sport-specific empty frame templates."""

from __future__ import annotations

import pytest

from pipeline.objects.analysis import Analysis
from pipeline.objects.frame import Frame
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_frame import TennisFrame


class _BareAnalysis(Analysis[Frame]):
    frame_class = Frame


class _FakeVideoReader:
    frame_count = 3

    def info(self) -> dict:
        return {
            "video_path": "fake.mp4",
            "frame_count": 3,
            "fps": 30.0,
            "height": 48,
            "width": 64,
        }


@pytest.mark.unit
def test_frame_empty_is_sport_agnostic() -> None:
    assert Frame.empty(2) == {"id": 2}


@pytest.mark.unit
def test_tennis_frame_empty_includes_ball_and_players() -> None:
    assert TennisFrame.empty(1) == {"id": 1, "players": [], "ball": None}


@pytest.mark.unit
def test_bare_analysis_from_video_has_no_tennis_keys() -> None:
    analysis = _BareAnalysis.from_video(
        _FakeVideoReader(),  # type: ignore[arg-type]
        "generic",
    )
    assert len(analysis.frames) == 3
    assert analysis.data["frames"][0] == {"id": 0}
    assert "ball" not in analysis.data["frames"][0]
    assert "players" not in analysis.data["frames"][0]


@pytest.mark.unit
def test_tennis_analysis_from_video_has_tennis_keys() -> None:
    analysis = TennisAnalysis.from_video(
        _FakeVideoReader(),  # type: ignore[arg-type]
        "tennis",
    )
    assert analysis.data["frames"][0] == {"id": 0, "players": [], "ball": None}
