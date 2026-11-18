"""Unit tests for CourtSchema, Analysis.court_schema, and legs.court."""

from __future__ import annotations

import pytest

from pipeline.objects.player_at_frame import PlayerAtFrame
from pipeline.objects.position import CourtPosition
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_court_schema import N_POINTS, TennisCourtSchema, model_xy
from pipeline.objects.tennis.tennis_frame import TennisFrame
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.stats.tennis.tennis_legs_court_stat import TennisLegsCourtStat


@pytest.mark.unit
def test_estimate_homography_identity_scale() -> None:
    schema = TennisCourtSchema()
    offset = (100.0, 200.0)
    keypoints = []
    confidences = []
    for pid in range(N_POINTS):
        mx, my = model_xy(pid)
        keypoints.append([mx + offset[0], my + offset[1]])
        confidences.append(1.0)

    h_mat = schema.estimate_homography(keypoints, confidences, min_conf=0.3)
    assert h_mat is not None

    pid = 5
    mx, my = model_xy(pid)
    img_pt = (mx + offset[0], my + offset[1])
    court = schema.image_to_court(h_mat, [img_pt])[0]
    assert court[0] == pytest.approx(mx, abs=1e-3)
    assert court[1] == pytest.approx(my, abs=1e-3)


@pytest.mark.unit
def test_estimate_homography_too_few_points() -> None:
    schema = TennisCourtSchema()
    keypoints = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (N_POINTS - 3)
    confidences = [1.0, 1.0, 1.0] + [0.0] * (N_POINTS - 3)
    assert schema.estimate_homography(keypoints, confidences, min_conf=0.5) is None


@pytest.mark.unit
def test_analysis_court_schema_instance() -> None:
    analysis = TennisAnalysis(
        {
            "sport": "tennis",
            "video": {"video_path": "x", "frame_count": 0, "fps": 1, "height": 1, "width": 1},
            "frames": [],
        }
    )
    assert isinstance(analysis.court_schema, TennisCourtSchema)
    assert analysis.court_schema.n_points == 14


@pytest.mark.unit
def test_legs_image_computed_not_in_json() -> None:
    player = PlayerAtFrame.create(position=[10.0, 20.0, 30.0, 40.0], confidence=0.9)
    assert player.legs.image is not None
    assert player.legs.x == pytest.approx(20.0)
    assert player.legs.y == pytest.approx(40.0)
    assert player.legs.court is None
    assert "legs" not in player.data


@pytest.mark.unit
def test_legs_court_setter_persists_json() -> None:
    player = PlayerAtFrame.create(position=[10.0, 20.0, 30.0, 40.0], confidence=0.9)
    player.legs.court = CourtPosition.create(x=1.2, y=-4.5)
    assert player.data["legs"]["court"] == {"x": 1.2, "y": -4.5}
    assert player.legs.court is not None
    assert player.legs.court.distance == pytest.approx((1.2**2 + 4.5**2) ** 0.5)


@pytest.mark.unit
def test_court_position_from_image_and_distance() -> None:
    schema = TennisCourtSchema()
    offset = (100.0, 200.0)
    keypoints = []
    confidences = []
    for pid in range(N_POINTS):
        mx, my = model_xy(pid)
        keypoints.append([mx + offset[0], my + offset[1]])
        confidences.append(1.0)
    h_mat = schema.estimate_homography(keypoints, confidences)
    assert h_mat is not None

    pos = CourtPosition.from_image((3.0 + offset[0], 4.0 + offset[1]), h_mat, schema)
    assert pos.x == pytest.approx(3.0, abs=1e-2)
    assert pos.y == pytest.approx(4.0, abs=1e-2)
    assert pos.distance == pytest.approx(5.0, abs=1e-2)


@pytest.mark.unit
def test_tennis_legs_court_stat() -> None:
    schema = TennisCourtSchema()
    offset = (100.0, 200.0)
    keypoints = []
    confidences = []
    for pid in range(N_POINTS):
        mx, my = model_xy(pid)
        keypoints.append([mx + offset[0], my + offset[1]])
        confidences.append(1.0)
    h_mat = schema.estimate_homography(keypoints, confidences)
    assert h_mat is not None

    feet_image = (3.0 + offset[0], 4.0 + offset[1])
    x0, x1 = feet_image[0] - 10.0, feet_image[0] + 10.0
    y1 = feet_image[1]
    y0 = y1 - 50.0

    frame_data = TennisFrame.empty(0)
    homo = TennisHomography.create(keypoints=keypoints, confidences=confidences)
    homo.matrix = h_mat.tolist()
    frame_data["homography"] = homo.data
    frame_data["players"] = [
        TennisPlayerAtFrame.create(position=[x0, y0, x1, y1], confidence=0.95).data
    ]

    analysis = TennisAnalysis(
        {
            "sport": "tennis",
            "video": {"video_path": "x", "frame_count": 1, "fps": 1, "height": 1, "width": 1},
            "frames": [frame_data],
        }
    )

    class _UnusedVideoReader:
        pass

    TennisLegsCourtStat({}).run(_UnusedVideoReader(), analysis)  # type: ignore[arg-type]

    player = analysis.frames[0].players[0]
    assert player.legs.court is not None
    assert player.legs.court.x == pytest.approx(3.0, abs=1e-2)
    assert player.legs.court.y == pytest.approx(4.0, abs=1e-2)
    assert player.legs.court.distance == pytest.approx(5.0, abs=1e-2)
    assert "stats" not in analysis.data
    assert player.data["legs"]["court"]["x"] == pytest.approx(3.0, abs=1e-2)
