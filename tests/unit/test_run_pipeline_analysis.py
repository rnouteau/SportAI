"""Unit tests for config-driven Analysis loading in run_pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.run import run_pipeline
from tests.helpers import make_synthetic_video


@pytest.mark.unit
def test_run_pipeline_requires_analysis_key(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=2)
    with pytest.raises(ValueError, match="analysis"):
        run_pipeline("tennis", str(video_path), {"models": []}, test=True)


@pytest.mark.unit
def test_run_pipeline_loads_analysis_from_config(tmp_path: Path) -> None:
    video_path = make_synthetic_video(tmp_path / "clip.mp4", n_frames=3)
    data = run_pipeline(
        "tennis",
        str(video_path),
        {
            "analysis": "pipeline.objects.tennis.TennisAnalysis",
            "models": [],
        },
        test=True,
    )
    assert data is not None
    assert data["sport"] == "tennis"
    assert len(data["frames"]) == 3
