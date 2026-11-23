"""TennisLegsCourtStat: the feet of every player projected on the court (metres)."""

from __future__ import annotations

import pytest

from pipeline.objects.tennis.tennis_court_schema import N_POINTS, TennisCourtSchema
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_legs_court_stat import TennisLegsCourtStat
from tests.helpers import COURT_OFFSET, make_tennis_analysis, shifted_court_keypoints


def _homography() -> TennisHomography:
    keypoints = shifted_court_keypoints()
    matrix = TennisCourtSchema().estimate_homography(keypoints, [1.0] * N_POINTS)
    assert matrix is not None
    return TennisHomography(keypoints=keypoints, matrix=matrix.tolist())


def _player_with_feet_at(x: float, y: float) -> TennisPlayerAtFrame:
    return TennisPlayerAtFrame(bbox=[x - 10.0, y - 50.0, x + 10.0, y], confidence=0.95)


def test_feet_are_projected_on_the_court() -> None:
    analysis = make_tennis_analysis(n_frames=1)
    frame = analysis.frames[0]
    frame.homography = _homography()
    frame.players = [_player_with_feet_at(3.0 + COURT_OFFSET[0], 4.0 + COURT_OFFSET[1])]

    TennisLegsCourtStat({}).run(None, analysis)  # type: ignore[arg-type]

    court = frame.players[0].court
    assert court is not None
    assert (court.x, court.y) == pytest.approx((3.0, 4.0), abs=1e-2)
    assert frame.players[0].to_dict()["legs"]["court"]["x"] == pytest.approx(3.0, abs=1e-2)


def test_frames_without_matrix_and_players_without_box_are_skipped() -> None:
    analysis = make_tennis_analysis(n_frames=2)
    analysis.frames[0].players = [_player_with_feet_at(10.0, 10.0)]  # no homography
    analysis.frames[1].homography = _homography()
    analysis.frames[1].players = [TennisPlayerAtFrame(confidence=0.9)]  # no box

    TennisLegsCourtStat({}).run(None, analysis)  # type: ignore[arg-type]

    assert analysis.frames[0].players[0].court is None
    assert analysis.frames[1].players[0].court is None


def test_without_court_schema_nothing_is_written() -> None:
    analysis = make_tennis_analysis(n_frames=1)
    analysis.court_schema = None
    analysis.frames[0].homography = _homography()
    analysis.frames[0].players = [_player_with_feet_at(103.0, 204.0)]

    TennisLegsCourtStat({}).run(None, analysis)  # type: ignore[arg-type]

    assert analysis.frames[0].players[0].court is None
