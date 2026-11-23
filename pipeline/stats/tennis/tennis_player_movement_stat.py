"""TennisPlayerMovementStat: distance run and speed of each selected player, in court metres.

Why not just add up the frame-to-frame steps? The box of a player moves by a few
pixels on every frame even when he stands still, so his court position jitters by
3-5 cm per frame. Summed over thousands of frames this doubles the distance, and
5 cm in 0.02 s (50 fps) already reads as 9 km/h. So, per player (``player_id``):

1. ``collect_tracks``: the court positions ``(frame, x, y)`` (frames without one are skipped);
2. ``split_at_gaps``: the track is cut where the player is missing for more than
   ``max_gap_seconds`` (cut-away, close-up): the straight line across the gap is not a run;
3. ``smooth``: centred moving average over ``smoothing_seconds``, removes the jitter;
4. distance = length of the smoothed segments; speed = displacement over
   ``speed_window_seconds``; mean speed = distance / time the player was tracked.

Output: ``distance_m``, ``mean_speed_kmh`` and ``max_speed_kmh`` of every player of
``analysis.final_stats.players``. Needs ``TennisLegsCourtStat`` (``player.court``) and
``TennisPlayerSelectorStat`` (``player_id`` and the final players) before it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields

from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader

DEFAULT_FPS = 30.0
MS_TO_KMH = 3.6

Point = tuple[int, float, float]  # frame id, court x (m), court y (m)


@dataclass(frozen=True)
class MovementConfig:
    smoothing_seconds: float = 0.5  # width of the moving average
    speed_window_seconds: float = 0.5  # speed = displacement over this duration
    max_gap_seconds: float = 0.5  # longer absence = the track is cut there

    @classmethod
    def from_dict(cls, config: dict | None) -> MovementConfig:
        config = config or {}
        return cls(**{f.name: float(config[f.name]) for f in fields(cls) if f.name in config})


@dataclass(frozen=True)
class Movement:
    distance_m: float
    duration_s: float  # time the player was tracked (sum of the segments)
    max_speed_kmh: float | None  # None when no segment is longer than the speed window

    @property
    def mean_speed_kmh(self) -> float | None:
        if self.duration_s <= 0:
            return None
        return self.distance_m / self.duration_s * MS_TO_KMH


def collect_tracks(analysis: TennisAnalysis) -> dict[int, list[Point]]:
    """``{player_id: [(frame, x, y), ...]}`` in frame order, for players with a court position."""
    tracks: dict[int, list[Point]] = defaultdict(list)
    for frame in analysis.frames:
        for player in frame.players:
            if player.player_id is not None and player.court is not None:
                tracks[player.player_id].append((frame.id, player.court.x, player.court.y))
    return dict(tracks)


def split_at_gaps(points: list[Point], max_gap_frames: int) -> list[list[Point]]:
    """Cut the track wherever two consecutive positions are more than ``max_gap_frames`` apart."""
    segments: list[list[Point]] = []
    for point in points:
        if segments and point[0] - segments[-1][-1][0] <= max_gap_frames:
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def smooth(points: list[Point], half_window_frames: int) -> list[Point]:
    """Centred moving average over ``half_window_frames`` on each side.

    Near both ends of a segment the window shrinks so as to stay centred (at the very
    first and last positions it is the position itself): a lopsided window would pull
    the ends inwards and shorten the distance of a player who keeps moving.
    """
    if half_window_frames <= 0 or len(points) < 3:
        return list(points)
    first, last = points[0][0], points[-1][0]
    smoothed: list[Point] = []
    start = end = 0
    for frame, _, _ in points:
        half = min(half_window_frames, frame - first, last - frame)
        while points[start][0] < frame - half:
            start += 1
        while end < len(points) and points[end][0] <= frame + half:
            end += 1
        window = [p for p in points[start:end] if p[0] >= frame - half]
        smoothed.append(
            (
                frame,
                sum(p[1] for p in window) / len(window),
                sum(p[2] for p in window) / len(window),
            )
        )
    return smoothed


def path_length(points: list[Point]) -> float:
    """Metres along the track."""
    return sum(math.dist(a[1:], b[1:]) for a, b in zip(points, points[1:], strict=False))


def speeds_kmh(points: list[Point], window_frames: int, fps: float) -> list[float]:
    """Speed between each position and the first one at least ``window_frames`` later."""
    speeds = []
    j = 0
    for i, (frame, x, y) in enumerate(points):
        j = max(j, i + 1)
        while j < len(points) and points[j][0] - frame < window_frames:
            j += 1
        if j == len(points):
            break
        seconds = (points[j][0] - frame) / fps
        speeds.append(math.dist((x, y), points[j][1:]) / seconds * MS_TO_KMH)
    return speeds


def player_movement(points: list[Point], fps: float, config: MovementConfig) -> Movement:
    distance = duration = 0.0
    speeds: list[float] = []
    max_gap = max(1, round(config.max_gap_seconds * fps))
    half_window = round(config.smoothing_seconds * fps / 2)
    speed_window = max(1, round(config.speed_window_seconds * fps))
    for segment in split_at_gaps(points, max_gap):
        smoothed = smooth(segment, half_window)
        distance += path_length(smoothed)
        duration += (segment[-1][0] - segment[0][0]) / fps
        speeds += speeds_kmh(smoothed, speed_window, fps)
    return Movement(distance, duration, max(speeds) if speeds else None)


class TennisPlayerMovementStat(Stat[TennisAnalysis]):
    """Distance and speed of the selected players (see the module docstring).

    Config keys (all optional), see ``MovementConfig``: smoothing_seconds,
    speed_window_seconds, max_gap_seconds.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.movement_config = MovementConfig.from_dict(config)

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        players = analysis.final_stats.players
        if not players:
            logger.warning("No final players (TennisPlayerSelectorStat missing?): no movement stat")
            return
        fps = float((analysis.video or {}).get("fps") or DEFAULT_FPS)
        tracks = collect_tracks(analysis)
        for player in players:
            points = tracks.get(player.id, [])
            if len(points) < 2:
                logger.warning(f"Player {player.id}: no court positions, no movement stat")
                continue
            movement = player_movement(points, fps, self.movement_config)
            player.distance_m = movement.distance_m
            player.mean_speed_kmh = movement.mean_speed_kmh
            player.max_speed_kmh = movement.max_speed_kmh
            logger.info(
                f"Player {player.id}: {movement.distance_m:.1f} m in {movement.duration_s:.1f} s,"
                f" max {movement.max_speed_kmh or 0:.1f} km/h"
            )
