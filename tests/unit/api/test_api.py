"""The API on a temporary outputs folder: list, summary, files, unknown and unsafe ids."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.analyses import file_name, is_valid_id, list_analyses, load_summary
from api.main import create_app

SAMPLE = Path(__file__).resolve().parents[2] / "fixtures" / "tennis_analysis_sample.json"


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    """``demo/`` = stats + overlay + heatmap of player 1; ``bare/`` = stats only; ``empty/``."""
    for name in ("demo", "bare"):
        (tmp_path / name).mkdir()
        shutil.copy(SAMPLE, tmp_path / name / "stats.json")
    (tmp_path / "demo" / "overlay.mp4").write_bytes(b"fake video")
    (tmp_path / "demo" / "heatmaps").mkdir()
    (tmp_path / "demo" / "heatmaps" / "heatmap_player_1.png").write_bytes(b"fake png")
    (tmp_path / "empty").mkdir()
    (tmp_path / "loose.json").write_text("{}")
    return tmp_path


@pytest.fixture
def client(outputs: Path) -> TestClient:
    return TestClient(create_app(outputs))


def test_is_valid_id() -> None:
    assert is_valid_id("fonseca-vs-djokovic_cut_3354")
    for bad in ("", "..", "../x", "a/b", "a\\b", "a b", "é"):
        assert not is_valid_id(bad)


def test_file_name_of_windows_and_posix_paths() -> None:
    assert file_name(".\\datasets\\clip one.mp4") == "clip one.mp4"
    assert file_name("datasets/clip.mp4") == "clip.mp4"


def test_list_only_folders_with_stats(client: TestClient) -> None:
    response = client.get("/api/analyses")
    assert response.status_code == 200
    assert [a["id"] for a in response.json()] == ["bare", "demo"]


def test_summary(client: TestClient) -> None:
    response = client.get("/api/analyses/demo")
    assert response.status_code == 200
    summary = response.json()
    assert summary["video_name"].endswith(".mp4")
    assert summary["fps"] > 0 and summary["frame_count"] > 0
    assert summary["duration_s"] == pytest.approx(summary["frame_count"] / summary["fps"])
    assert summary["overlay_url"] == "/files/demo/overlay.mp4"
    assert [p["id"] for p in summary["players"]] == [1, 2]
    assert summary["players"][0]["heatmap_url"] == "/files/demo/heatmaps/heatmap_player_1.png"
    assert summary["players"][1]["heatmap_url"] is None  # not rendered
    assert "frames" not in summary  # the summary never carries the frames


def test_missing_files_give_null_urls(client: TestClient) -> None:
    summary = client.get("/api/analyses/bare").json()
    assert summary["overlay_url"] is None
    assert all(p["heatmap_url"] is None for p in summary["players"])


@pytest.mark.parametrize("analysis_id", ["unknown", "empty", "..%2F..", "%2E%2E"])
def test_unknown_or_unsafe_id_is_404(client: TestClient, analysis_id: str) -> None:
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_files_are_served(client: TestClient) -> None:
    response = client.get("/files/demo/overlay.mp4")
    assert response.status_code == 200
    assert response.content == b"fake video"
    assert client.get("/files/demo/missing.mp4").status_code == 404


def test_cors_allows_the_vite_dev_server(client: TestClient) -> None:
    response = client.get("/api/analyses", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_missing_outputs_folder(tmp_path: Path) -> None:
    assert list_analyses(tmp_path / "nope") == []
    assert load_summary(tmp_path, "nope") is None
    assert TestClient(create_app(tmp_path / "nope")).get("/api/analyses").json() == []
