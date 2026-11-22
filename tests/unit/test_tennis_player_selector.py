"""Unit tests for TennisPlayerSelectorStat (pick the two players, no court zone)."""

from __future__ import annotations

import math

import pytest

from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_final_stats import TennisFinalStats
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_player_selector_stat import (
    SelectorConfig,
    TennisPlayerSelectorStat,
    Track,
    lateral_factor,
    select_players,
)
from pipeline.stats.tennis.tennis_side_stat import TennisSideStat

FPS = 50.0
N = 200  # frames (4 s)

# (track_id, x, y) per frame, built by the helpers below
People = dict[int, list[tuple[int, float, float]]]


def _walk(track_id: int, start: tuple[float, float], end: tuple[float, float], frames: range):
    n = max(len(frames) - 1, 1)
    for k, f in enumerate(frames):
        t = k / n
        yield f, (track_id, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def _still(track_id: int, pos: tuple[float, float], frames: range, jitter: float = 0.05):
    for f in frames:
        dx = jitter if f % 2 else -jitter
        yield f, (track_id, pos[0] + dx, pos[1] - dx)


def _people(*streams) -> People:  # noqa: ANN002
    people: People = {}
    for stream in streams:
        for f, item in stream:
            people.setdefault(f, []).append(item)
    return people


def _analysis(people: People, n_frames: int = N, with_homography: bool = True) -> TennisAnalysis:
    frames = [{"id": i, "players": [], "ball": None} for i in range(n_frames)]
    analysis = TennisAnalysis({"sport": "tennis", "video": {"fps": FPS}, "frames": frames})
    for frame in analysis.frames:
        if with_homography:
            homo = TennisHomography.create()
            homo.matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            frame.homography = homo
        players = []
        for track_id, x, y in people.get(frame.id, []):
            player = TennisPlayerAtFrame.create(position=[0.0, 0.0, 10.0, 20.0], confidence=0.9)
            player.track_id = track_id
            player.legs.court = CourtPosition.create(x=x, y=y)
            players.append(player)
        frame.players = players
    return analysis


def _run(analysis: TennisAnalysis, config: dict | None = None) -> TennisAnalysis:
    TennisSideStat().run(None, analysis)  # type: ignore[arg-type]
    TennisPlayerSelectorStat(config or {}).run(None, analysis)  # type: ignore[arg-type]
    return analysis


def _track(track_id: int, positions: dict[int, tuple[float, float]]) -> Track:
    sides = {f: ("near" if y < 0 else "far") for f, (_, y) in positions.items()}
    return Track(track_id, positions, sides)


def _kept(analysis: TennisAnalysis) -> dict[int, set[tuple[int, int]]]:
    """{frame: {(track_id, player_id), ...}}"""
    return {
        f.id: {(p.track_id, p.player_id) for p in f.players}  # type: ignore[misc]
        for f in analysis.frames
    }


# A typical point: both players run, line judge and ball kid barely move.
NEAR_PLAYER = list(_walk(1, (-3.0, -12.0), (4.0, -13.0), range(N)))
FAR_PLAYER = list(_walk(2, (3.0, 12.5), (-4.0, 11.0), range(N)))
LINE_JUDGE = list(_still(3, (6.4, 11.9), range(N)))  # far side, at the corner
BALL_KID_NET = list(_still(4, (-6.4, 0.5), range(N)))  # crouched at the net post
CROSSING_KID = list(_walk(5, (2.0, -4.0), (2.0, 4.0), range(0, 60)))  # crosses the net line


@pytest.mark.unit
def test_path_length_is_not_inflated_by_jitter() -> None:
    track = Track(9, {f: (x, y) for f, (_, x, y) in _still(9, (0.0, 0.0), range(N))})
    per_frame = sum(math.dist(track.positions[f], track.positions[f + 1]) for f in range(N - 1))
    assert per_frame > 10.0  # summing every frame: a still person "walks" 10+ m
    assert track.path_length(step_frames=25) < 0.1


@pytest.mark.unit
def test_path_length_of_a_walk() -> None:
    track = Track(1, {f: (x, y) for f, (_, x, y) in _walk(1, (0.0, 0.0), (6.0, 8.0), range(N))})
    assert track.path_length(step_frames=25) == pytest.approx(10.0, rel=0.15)


@pytest.mark.unit
def test_side_needs_a_clear_majority() -> None:
    near = _track(1, {0: (0.0, -5.0), 1: (0.0, -6.0)})
    far = _track(2, {0: (0.0, 5.0), 1: (0.0, 6.0)})
    mixed = _track(3, {0: (0.0, -1.0), 1: (0.0, 1.0)})
    unknown = Track(4, {0: (0.0, 5.0)})  # TennisSideStat did not run
    assert (near.side(0.8), far.side(0.8), mixed.side(0.8)) == ("near", "far", None)
    assert unknown.side(0.8) is None


@pytest.mark.unit
def test_keeps_one_player_per_side_and_drops_officials() -> None:
    analysis = _run(
        _analysis(_people(NEAR_PLAYER, FAR_PLAYER, LINE_JUDGE, BALL_KID_NET, CROSSING_KID))
    )
    for frame_id, kept in _kept(analysis).items():
        assert kept == {(1, 1), (2, 2)}, frame_id
    stats = TennisFinalStats(analysis.final_stats)
    assert [(p.id, p.name, p.track_ids) for p in stats.players] == [
        (1, "player A", [1]),
        (2, "player B", [2]),
    ]


@pytest.mark.unit
def test_player_far_outside_the_court_is_kept() -> None:
    """No zone: a player chasing a ball 20 m behind the baseline stays selected."""
    near = _walk(1, (0.0, -12.0), (14.0, -20.0), range(N))
    analysis = _run(_analysis(_people(near, FAR_PLAYER, LINE_JUDGE)))
    assert all((1, 1) in kept for kept in _kept(analysis).values())


@pytest.mark.unit
def test_split_track_is_linked_back_to_the_player() -> None:
    """The tracker lost the far player at frame 100 and gave him id 8."""
    far_a = _walk(2, (3.0, 12.0), (0.0, 12.0), range(0, 100))
    far_b = _walk(8, (0.2, 12.0), (-3.0, 11.0), range(105, N))
    late_stranger = _walk(9, (-6.0, 5.0), (-6.0, 9.0), range(150, N))  # overlaps far_b
    analysis = _run(_analysis(_people(NEAR_PLAYER, far_a, far_b, late_stranger)))
    kept = _kept(analysis)
    assert kept[50] == {(1, 1), (2, 2)}
    assert kept[150] == {(1, 1), (8, 2)}  # id 8 continues player 2, id 9 dropped
    far = TennisFinalStats(analysis.final_stats).players[1]
    assert far.track_ids == [2, 8]


@pytest.mark.unit
def test_split_track_not_linked_when_gap_or_jump_too_big() -> None:
    far_a = _walk(2, (3.0, 12.0), (0.0, 12.0), range(0, 80))
    far_jump = _walk(8, (6.0, 4.0), (5.0, 3.0), range(85, 140))  # 8 m away from id 2's end
    tracks: dict[int, Track] = {}
    for f, (tid, x, y) in [*far_a, *far_jump]:
        track = tracks.setdefault(tid, Track(tid))
        track.positions[f] = (x, y)
        track.sides[f] = "far"
    selection = select_players(tracks, N, FPS, SelectorConfig())
    assert selection["far"] == [2]


@pytest.mark.unit
def test_frames_without_court_keep_selected_players_only() -> None:
    analysis = _analysis(_people(NEAR_PLAYER, FAR_PLAYER, LINE_JUDGE))
    frame = analysis.frames[10]
    frame.homography = None
    for player in frame.players:  # no projection on this frame
        player.legs.court = None
    _run(analysis)
    assert {p.track_id for p in analysis.frames[10].players} == {1, 2}


@pytest.mark.unit
def test_without_court_positions_players_are_left_unchanged() -> None:
    analysis = _analysis(_people(NEAR_PLAYER, FAR_PLAYER), with_homography=False)
    for frame in analysis.frames:
        for player in frame.players:
            player.legs.court = None
    before = [len(f.players) for f in analysis.frames]
    _run(analysis)
    assert [len(f.players) for f in analysis.frames] == before
    assert all(p.player_id is None for f in analysis.frames for p in f.players)


@pytest.mark.unit
def test_config_values_are_read() -> None:
    stat = TennisPlayerSelectorStat({"min_side_ratio": 0.9, "max_link_distance_m": 5})
    assert stat.selector_config.min_side_ratio == 0.9
    assert stat.selector_config.max_link_distance_m == 5.0


def _jittery(track_id: int, pos: tuple[float, float], frames: range, amplitude: float):
    """Someone in place whose projected feet jump around (e.g. the chair umpire)."""
    for f in frames:
        k = (f // 10) % 4
        dx, dy = [(amplitude, 0.0), (0.0, amplitude), (-amplitude, 0.0), (0.0, -amplitude)][k]
        yield f, (track_id, pos[0] + dx, pos[1] + dy)


@pytest.mark.unit
def test_lateral_factor() -> None:
    assert lateral_factor(4.0, 6.5, 10.0) == 1.0
    assert lateral_factor(8.25, 6.5, 10.0) == pytest.approx(0.5)
    assert lateral_factor(12.0, 6.5, 10.0) == 0.0


@pytest.mark.unit
def test_split_player_beats_a_continuous_ball_kid() -> None:
    """Real case (Fonseca clip): the far player is ids 2 then 55, overlapping a few frames;
    a ball kid walking along the back fence is one id for the whole clip."""
    far_a = list(_walk(2, (-4.0, 16.0), (-1.0, 15.0), range(0, 105)))
    far_b = list(_walk(55, (-1.1, 15.0), (2.0, 17.0), range(100, N)))  # 5 frames overlap
    ball_kid = list(_walk(3, (-6.0, 20.5), (0.0, 20.5), range(N)))  # 6 m along the fence
    analysis = _run(_analysis(_people(NEAR_PLAYER, far_a, far_b, ball_kid)))
    far = TennisFinalStats(analysis.final_stats).players[1]
    assert far.track_ids == [2, 55]
    kept = _kept(analysis)
    assert kept[50] == {(1, 1), (2, 2)}
    assert kept[102] == {(1, 1), (2, 2)} or kept[102] == {(1, 1), (55, 2)}  # one box per player
    assert kept[150] == {(1, 1), (55, 2)}


@pytest.mark.unit
def test_chair_umpire_is_not_taken_for_the_far_player() -> None:
    """The umpire sits beside the net, high up: his projected feet jump around."""
    umpire = list(_jittery(10, (-8.6, 6.9), range(N), amplitude=1.5))
    analysis = _run(_analysis(_people(NEAR_PLAYER, FAR_PLAYER, umpire)))
    assert TennisFinalStats(analysis.final_stats).players[1].track_ids == [2]


@pytest.mark.unit
def test_without_side_stat_players_are_left_unchanged() -> None:
    analysis = _analysis(_people(NEAR_PLAYER, FAR_PLAYER, LINE_JUDGE))
    TennisPlayerSelectorStat({}).run(None, analysis)  # type: ignore[arg-type]
    assert all(len(f.players) == 3 for f in analysis.frames)
