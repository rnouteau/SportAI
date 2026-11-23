"""Shared pytest setup: ``--update-baseline``, markers from the folder, common fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers import make_synthetic_video

TESTS_DIR = Path(__file__).resolve().parent
# Tests get the marker of their folder under tests/: ``pytest -m unit`` / ``-m integration``.
FOLDER_MARKERS = {"unit", "integration"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--update-baseline", action="store_true", help="Rewrite the NRT baselines")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path)).resolve()
        if not path.is_relative_to(TESTS_DIR):
            continue
        folder = path.relative_to(TESTS_DIR).parts[0]
        if folder in FOLDER_MARKERS:
            item.add_marker(folder)


@pytest.fixture
def update_baseline(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-baseline"))


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Callable[..., Path]:
    """``synthetic_video(n_frames=5, name="clip.mp4", size=(64, 48))`` → path of a tiny mp4."""

    def make(n_frames: int = 5, name: str = "clip.mp4", size: tuple[int, int] = (64, 48)) -> Path:
        return make_synthetic_video(tmp_path / name, n_frames=n_frames, size=size)

    return make
