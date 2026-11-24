"""Read the analyses written by the pipeline (one folder each). No FastAPI here: plain
functions, easy to test.

    <root>/<analysis id>/stats.json
                        /overlay.mp4                  (optional)
                        /heatmaps/heatmap_player_<id>.png  (optional)
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from api.schemas import AnalysisSummary, PlayerSummary
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.run import HEATMAPS_DIR, OVERLAY_FILE, STATS_FILE

FILES_URL = "/files"  # where main.py mounts the outputs root
_VALID_ID = re.compile(r"[A-Za-z0-9_-]+")


def is_valid_id(analysis_id: str) -> bool:
    """Only letters, digits, ``_`` and ``-``: an id can never leave the outputs root
    (``..``, ``/`` or ``\\`` are refused)."""
    return bool(_VALID_ID.fullmatch(analysis_id))


def file_name(path: str) -> str:
    """Last part of a path written on any OS (``Path.name`` keeps Windows ``\\`` on Linux)."""
    return re.split(r"[\\/]", path)[-1]


def file_url(root: Path, analysis_id: str, relative: str) -> str | None:
    """URL of a file of the analysis, ``None`` when the file does not exist."""
    if not (root / analysis_id / relative).is_file():
        return None
    return f"{FILES_URL}/{analysis_id}/{relative}"


def load_summary(root: Path, analysis_id: str) -> AnalysisSummary | None:
    """Summary of one analysis (video + final players, not the frames), ``None`` if unknown."""
    if not is_valid_id(analysis_id):
        return None
    stats_path = root / analysis_id / STATS_FILE
    if not stats_path.is_file():
        return None

    analysis = TennisAnalysis.load_json(stats_path)
    video = analysis.video or {}
    fps = float(video.get("fps") or 0.0)
    frame_count = int(video.get("frame_count") or len(analysis.frames))
    players = [
        PlayerSummary(
            id=player.id,
            name=player.name,
            distance_m=player.distance_m,
            mean_speed_kmh=player.mean_speed_kmh,
            max_speed_kmh=player.max_speed_kmh,
            heatmap_url=file_url(
                root, analysis_id, f"{HEATMAPS_DIR}/heatmap_player_{player.id}.png"
            ),
        )
        for player in analysis.final_stats.players
    ]
    return AnalysisSummary(
        id=analysis_id,
        video_name=file_name(str(video.get("video_path") or analysis_id)),
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps if fps > 0 else 0.0,
        overlay_url=file_url(root, analysis_id, OVERLAY_FILE),
        players=players,
    )


def list_analyses(root: Path) -> list[AnalysisSummary]:
    """Every sub-folder of ``root`` holding a ``stats.json``, sorted by id."""
    if not root.is_dir():
        logger.warning(f"Outputs folder {root} does not exist: no analysis")
        return []
    summaries = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = load_summary(root, folder.name)
        if summary is not None:
            summaries.append(summary)
    return summaries
