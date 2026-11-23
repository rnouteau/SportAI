"""Pick the two singles players among everyone the tracker follows — without a court zone.

The decision is taken per tracker id over the whole video, from the feet projected on
the court (``player.court``, from ``TennisLegsCourtStat``) and the side of the court on
each frame (``player.side``, from ``TennisSideStat``). Both must run first:

- **side**: share of the id's frames in the near or far half; it must be clear
  (``min_side_ratio``), so a ball kid crossing the court is no one's.
- **movement**: path length in metres between the mean positions of consecutive
  ``sample_seconds`` windows. Summing every frame would add up the jitter of the box
  (a still line judge would seem to run).
- **presence**: frames where the id is seen / frames that have a homography.

- **lateral position**: the *median* distance to the centre line. Officials sit or stand
  beside the court for the whole video (the chair umpire, high on his chair, even
  looks like he moves, because his feet are not on the ground). A player who runs
  wide for a few seconds keeps a central median, so he is not penalised. Depth is
  never limited: a player can stay 20 m behind his baseline.

The tracker often splits one player into several ids (lost, then found again with a new
id, sometimes with a short overlap). So ids of the same half are first **chained**: an id
continues another when it starts at most ``max_link_gap_seconds`` after the other ends,
within ``max_link_distance_m``, or overlaps it for at most ``max_link_overlap_seconds``
while staying within that distance. Chains, not single ids, are then scored:

``score = movement × presence × lateral factor``: the best chain of each half is a player.

Output: everyone else is removed from the frames; kept players get ``player_id``
(1 = near, 2 = far) and ``analysis.final_stats["players"]`` lists both players (id,
name, tracker ids).

Limit: ``player_id`` means "near / far player". Players change ends at changeovers, so
over a whole match the same id is not always the same person (that needs re-identification).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player import TennisPlayer
from pipeline.objects.tennis.tennis_player_at_frame import FAR, NEAR, TennisPlayerAtFrame
from pipeline.stats.stat import Stat
from pipeline.video_reader import VideoReader

PLAYER_IDS = {NEAR: 1, FAR: 2}
PLAYER_NAME = {NEAR: "player A", FAR: "player B"}
DEFAULT_FPS = 25.0


@dataclass
class Track:
    """Court positions and sides of one tracker id (frames with a projected position)."""

    track_id: int
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    sides: dict[int, str] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return sorted(self.positions)

    @property
    def first(self) -> int:
        return min(self.positions)

    @property
    def last(self) -> int:
        return max(self.positions)

    def median_abs_x(self) -> float:
        """Median distance to the centre line (metres)."""
        values = sorted(abs(x) for x, _ in self.positions.values())
        if not values:
            return 0.0
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    def near_ratio(self) -> float:
        """Share of the frames with a side where the id is in the near half."""
        sides = list(self.sides.values())
        return sum(1 for side in sides if side == NEAR) / len(sides) if sides else 0.0

    def side(self, min_side_ratio: float) -> str | None:
        if not self.sides:
            return None
        ratio = self.near_ratio()
        if ratio >= min_side_ratio:
            return NEAR
        if ratio <= 1.0 - min_side_ratio:
            return FAR
        return None

    def path_length(self, step_frames: int) -> float:
        """Metres walked between the mean positions of consecutive ``step_frames`` windows.

        Averaging each window cancels the frame-to-frame jitter of the box, which would
        otherwise add up to metres for someone standing still.
        """
        windows: dict[int, list[tuple[float, float]]] = {}
        for frame, pos in self.positions.items():
            windows.setdefault(frame // step_frames, []).append(pos)
        means = [
            (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
            for _, pts in sorted(windows.items())
        ]
        return sum(math.dist(a, b) for a, b in zip(means, means[1:], strict=False))


def collect_tracks(analysis: TennisAnalysis) -> tuple[dict[int, Track], int]:
    """``({track_id: Track}, number of frames with a homography)``."""
    tracks: dict[int, Track] = {}
    frames_with_court = 0
    for frame in analysis.frames:
        homo = frame.homography
        if homo is not None and homo.matrix is not None:
            frames_with_court += 1
        for player in frame.players:
            court = player.court
            if player.track_id is None or court is None:
                continue
            track = tracks.setdefault(player.track_id, Track(player.track_id))
            track.positions[frame.id] = (court.x, court.y)
            if player.side is not None:
                track.sides[frame.id] = player.side
    return tracks, frames_with_court


@dataclass(frozen=True)
class SelectorConfig:
    sample_seconds: float = 0.5
    min_side_ratio: float = 0.8
    min_track_seconds: float = 0.5
    max_link_gap_seconds: float = 2.0
    max_link_distance_m: float = 3.0
    max_link_overlap_seconds: float = 0.5
    # Median |x| (metres from the centre line; doubles sideline = 5.485): full score
    # up to lateral_soft_m, then down to 0 at lateral_zero_m.
    lateral_soft_m: float = 6.5
    lateral_zero_m: float = 10.0

    @classmethod
    def from_dict(cls, config: dict) -> SelectorConfig:
        known = {k: float(v) for k, v in config.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def lateral_factor(median_abs_x: float, soft_m: float, zero_m: float) -> float:
    """1 up to ``soft_m`` from the centre line, then linearly down to 0 at ``zero_m``."""
    if median_abs_x <= soft_m:
        return 1.0
    if median_abs_x >= zero_m or zero_m <= soft_m:
        return 0.0
    return (zero_m - median_abs_x) / (zero_m - soft_m)


def link_cost(a: Track, b: Track, cfg: SelectorConfig, fps: float) -> float | None:
    """Distance (m) if ``b`` continues ``a`` (same person, new tracker id), else None."""
    if b.first <= a.first or b.last <= a.last:
        return None
    common = set(a.positions) & set(b.positions)
    if common:  # both ids on the same person for a few frames
        if len(common) > round(cfg.max_link_overlap_seconds * fps):
            return None
        dists = sorted(math.dist(a.positions[f], b.positions[f]) for f in common)
        dist = dists[len(dists) // 2]
    else:
        if b.first - a.last > round(cfg.max_link_gap_seconds * fps):
            return None
        dist = math.dist(a.positions[a.last], b.positions[b.first])
    return dist if dist <= cfg.max_link_distance_m else None


def build_chains(tracks: list[Track], cfg: SelectorConfig, fps: float) -> list[list[Track]]:
    """Group the tracks of one half into chains (one chain = one person over time)."""
    links = []
    for a in tracks:
        for b in tracks:
            cost = link_cost(a, b, cfg, fps)
            if cost is not None:
                links.append((cost, a.track_id, b.track_id))
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for _, a_id, b_id in sorted(links):  # closest continuations first
        if a_id not in successor and b_id not in predecessor:
            successor[a_id] = b_id
            predecessor[b_id] = a_id
    by_id = {t.track_id: t for t in tracks}
    chains = []
    for track in sorted(tracks, key=lambda t: t.first):
        if track.track_id in predecessor:
            continue
        chain = [track]
        while chain[-1].track_id in successor:
            chain.append(by_id[successor[chain[-1].track_id]])
        chains.append(chain)
    return chains


def merge(chain: list[Track]) -> Track:
    """One Track with the positions of every id of the chain."""
    merged = Track(chain[0].track_id)
    for track in chain:
        merged.positions.update(track.positions)
        merged.sides.update(track.sides)
    return merged


def chain_score(
    chain: list[Track], frames_with_court: int, fps: float, cfg: SelectorConfig
) -> float:
    track = merge(chain)
    if len(track.positions) < max(1, round(cfg.min_track_seconds * fps)):
        return 0.0
    presence = len(track.positions) / max(frames_with_court, 1)
    lateral = lateral_factor(track.median_abs_x(), cfg.lateral_soft_m, cfg.lateral_zero_m)
    return track.path_length(max(1, round(cfg.sample_seconds * fps))) * presence * lateral


def select_players(
    tracks: dict[int, Track], frames_with_court: int, fps: float, cfg: SelectorConfig
) -> dict[str, list[int]]:
    """``{"near": [track ids], "far": [track ids]}``: the best chain of each half."""
    selection: dict[str, list[int]] = {}
    for side in (NEAR, FAR):
        on_side = [t for t in tracks.values() if t.side(cfg.min_side_ratio) == side]
        chains = build_chains(on_side, cfg, fps)
        scored = [(chain_score(c, frames_with_court, fps, cfg), c) for c in chains]
        scored = [(score, c) for score, c in scored if score > 0.0]
        if scored:
            best = max(scored, key=lambda item: item[0])[1]
            selection[side] = [t.track_id for t in best]
    return selection


def apply_selection(analysis: TennisAnalysis, selection: dict[str, list[int]]) -> None:
    """Keep only selected players in every frame and give them their ``player_id``."""
    player_of = {tid: PLAYER_IDS[side] for side, tids in selection.items() for tid in tids}
    for frame in analysis.frames:
        best: dict[int, TennisPlayerAtFrame] = {}
        for player in frame.players:
            pid = player_of.get(player.track_id) if player.track_id is not None else None
            if pid is None:
                continue
            player.player_id = pid
            current = best.get(pid)
            if current is None or (player.confidence or 0.0) > (current.confidence or 0.0):
                best[pid] = player
        frame.players = [best[pid] for pid in sorted(best)]

    analysis.final_stats.players = sorted(
        (
            TennisPlayer(id=PLAYER_IDS[side], name=PLAYER_NAME[side], track_ids=tids)
            for side, tids in selection.items()
        ),
        key=lambda p: p.id,
    )


class TennisPlayerSelectorStat(Stat[TennisAnalysis]):
    """Keep the two singles players (one per half) and give them a stable ``player_id``.

    Config keys (all optional), see ``SelectorConfig``: sample_seconds, min_side_ratio,
    min_track_seconds, max_link_gap_seconds, max_link_distance_m, max_link_overlap_seconds,
    lateral_soft_m, lateral_zero_m.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.selector_config = SelectorConfig.from_dict(self.config)
        logger.info("TennisPlayerSelectorStat initialized")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisPlayerSelectorStat")
        tracks, frames_with_court = collect_tracks(analysis)
        if not tracks:
            logger.warning(
                "No player with a court position (court model or TennisLegsCourtStat missing?): "
                "players left unchanged"
            )
            return
        if not any(track.sides for track in tracks.values()):
            logger.warning("No player side (TennisSideStat missing?): players left unchanged")
            return
        fps = float((analysis.video or {}).get("fps") or DEFAULT_FPS)
        selection = select_players(tracks, frames_with_court, fps, self.selector_config)
        logger.info(f"Selected players: {selection}")
        apply_selection(analysis, selection)
