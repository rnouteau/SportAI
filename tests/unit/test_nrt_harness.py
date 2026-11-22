"""Unit tests for tests/nrt/harness.py (config preparation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.nrt import harness


@pytest.mark.unit
def test_prepare_config_uses_per_model_weights(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    config = {
        "models": [
            {"name": "player", "config": {"weight": "./models/player.pt"}},
            {
                "name": "court",
                "config": {"weight": "./models/court.pt"},
                "weights": {"repo_id": "r", "filename": "court/court.pt"},
            },
        ],
        "weights": {"repo_id": "r", "filename": "player/player.pt"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(
        harness, "resolve_weights", lambda model, weights: calls.append(weights["filename"]) or "w"
    )
    monkeypatch.delenv("SPORTAI_DEVICE", raising=False)

    harness.prepare_config(path)

    assert calls == ["player/player.pt", "court/court.pt"]
