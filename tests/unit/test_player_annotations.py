"""Unit tests for the player annotation model (tools/player_annotator/player_annotations.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tools.player_annotator.player_annotations import (
    UNKNOWN_ID,
    PlayerAnnotations,
    PlayerBox,
    TrackIdMapper,
)


def _box(tid: int, x: float, y: float, w: float = 20, h: float = 50, kf: bool = True) -> PlayerBox:
    return PlayerBox.from_corners(tid, x, y, x + w, y + h, keyframe=kf)


@pytest.mark.unit
def test_from_corners_normalises_reversed_drag() -> None:
    box = PlayerBox.from_corners(1, 50, 80, 10, 20)
    assert (box.x0, box.y0, box.x1, box.y1) == (10, 20, 50, 80)


@pytest.mark.unit
def test_set_box_replaces_same_id_and_marks_reviewed() -> None:
    ann = PlayerAnnotations()
    ann.set_box(3, _box(1, 0, 0))
    ann.set_box(3, _box(1, 100, 100))
    ann.set_box(3, _box(2, 300, 300))
    boxes = ann.frame_boxes(3)
    assert sorted(b.track_id for b in boxes) == [1, 2]
    assert next(b for b in boxes if b.track_id == 1).x0 == 100
    assert ann.is_reviewed(3)


@pytest.mark.unit
def test_unknown_ids_can_coexist() -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, _box(UNKNOWN_ID, 0, 0))
    ann.set_box(0, _box(UNKNOWN_ID, 100, 0))
    assert len(ann.frame_boxes(0)) == 2


@pytest.mark.unit
def test_box_at_prefers_smallest_and_delete() -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, _box(1, 0, 0, 200, 200))
    ann.set_box(0, _box(2, 50, 50, 20, 20))
    hit = ann.box_at(0, 60, 60)
    assert hit is not None and hit.track_id == 2
    assert ann.delete_at(0, 60, 60)
    assert [b.track_id for b in ann.frame_boxes(0)] == [1]
    assert not ann.delete_at(0, 500, 500)


@pytest.mark.unit
def test_reassign_swaps_with_holder() -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, _box(1, 0, 0))
    ann.set_box(0, _box(2, 100, 0))
    assert ann.reassign_at(0, 5, 5, 2)
    ids = {b.x0: b.track_id for b in ann.frame_boxes(0)}
    assert ids == {0: 2, 100: 1}
    assert not ann.reassign_at(0, 999, 999, 1)


@pytest.mark.unit
def test_swap_ids_frame_and_to_end() -> None:
    ann = PlayerAnnotations()
    for f in range(5):
        ann.set_box(f, _box(1, 0, 0), review=False)
        ann.set_box(f, _box(2, 100, 0), review=False)
    assert ann.swap_ids(1) == 1
    assert ann.swap_ids(3, to_end=True) == 2
    first = {f: next(b.track_id for b in ann.frame_boxes(f) if b.x0 == 0) for f in range(5)}
    assert first == {0: 1, 1: 2, 2: 1, 3: 2, 4: 2}


@pytest.mark.unit
def test_copy_previous_uses_nearest_annotated_frame() -> None:
    ann = PlayerAnnotations()
    ann.set_box(2, _box(1, 10, 10))
    assert ann.copy_previous(7)
    copied = ann.frame_boxes(7)
    assert len(copied) == 1 and copied[0].x0 == 10 and not copied[0].keyframe
    assert not PlayerAnnotations().copy_previous(5)


@pytest.mark.unit
def test_interpolate_fills_between_keyframes() -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, _box(1, 0, 0))
    ann.set_box(4, _box(1, 40, 80, kf=False))
    ann.set_box(2, _box(1, 999, 999, kf=False), review=False)  # wrong pre-annotation
    written = ann.interpolate_to(4)
    assert written == 3
    mid = ann.frame_boxes(2)[0]
    assert (mid.x0, mid.y0) == pytest.approx((20.0, 40.0))
    assert not mid.keyframe
    assert ann.frame_boxes(4)[0].keyframe
    assert all(ann.is_reviewed(f) for f in range(5))


@pytest.mark.unit
def test_interpolate_without_previous_keyframe_only_marks_keyframe() -> None:
    ann = PlayerAnnotations()
    ann.set_box(5, _box(1, 0, 0, kf=False))
    assert ann.interpolate_to(5) == 0
    assert ann.frame_boxes(5)[0].keyframe


@pytest.mark.unit
def test_csv_round_trip_keeps_empty_reviewed_frames(tmp_path: Path) -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, _box(1, 0, 0))
    ann.set_box(1, _box(2, 5, 5, kf=False), review=False)
    ann.mark_reviewed(9)  # close-up, no player
    path = ann.save_csv(tmp_path / "clip_players.csv")

    loaded = PlayerAnnotations.load_csv(path)
    assert loaded.reviewed == {0, 9}
    assert loaded.frame_boxes(0)[0].keyframe
    assert loaded.frame_boxes(1)[0].track_id == 2

    # The detection evaluator reads the same file with dropna on coordinates.
    df = pd.read_csv(path).dropna(subset=["x0", "y0", "x1", "y1"])
    assert sorted(df["frame"].tolist()) == [0, 1]


@pytest.mark.unit
def test_legacy_csv_without_track_id(tmp_path: Path) -> None:
    path = tmp_path / "old_players.csv"
    pd.DataFrame([{"frame": 4, "class_id": 0, "x0": 30, "y0": 40, "x1": 10, "y1": 20}]).to_csv(
        path, index=False
    )
    ann = PlayerAnnotations.load_csv(path)
    box = ann.frame_boxes(4)[0]
    assert box.track_id == UNKNOWN_ID
    assert (box.x0, box.y0, box.x1, box.y1) == (10, 20, 30, 40)
    assert ann.is_reviewed(4)


@pytest.mark.unit
def test_mot_export_only_reviewed_and_known_ids(tmp_path: Path) -> None:
    ann = PlayerAnnotations()
    ann.set_box(0, PlayerBox.from_corners(2, 10, 20, 40, 120))
    ann.set_box(0, PlayerBox.from_corners(UNKNOWN_ID, 0, 0, 5, 5))
    ann.set_box(1, _box(1, 0, 0), review=False)
    gt_path, n_boxes, skipped = ann.export_mot(tmp_path / "clip_mot")
    assert (n_boxes, skipped) == (1, 1)
    assert gt_path.read_text().splitlines() == ["1,2,10.00,20.00,30.00,100.00,1,1,1.0"]
    assert (gt_path.parent / "labels.txt").read_text().strip() == "player"


@pytest.mark.unit
def test_mapper_first_frame_lowest_box_is_player_one() -> None:
    mapper = TrackIdMapper()
    boxes = mapper.update(
        [
            (7, 0.9, [500, 100, 540, 200]),  # far player (top of image)
            (3, 0.8, [600, 500, 660, 650]),  # near player (bottom)
            (9, 0.3, [10, 10, 20, 30]),  # ball kid, dropped (top-2 only)
        ]
    )
    ids = {b.track_id: b.y1 for b in boxes}
    assert ids == {1: 650, 2: 200}


@pytest.mark.unit
def test_mapper_keeps_ids_and_assigns_new_tracker_id_by_position() -> None:
    mapper = TrackIdMapper()
    mapper.update([(7, 0.9, [500, 100, 540, 200]), (3, 0.8, [600, 500, 660, 650])])
    # Tracker lost the far player and gave him a new id 12.
    boxes = mapper.update([(3, 0.9, [605, 505, 665, 655]), (12, 0.7, [510, 110, 550, 210])])
    ids = {b.track_id: b.y1 for b in boxes}
    assert ids == {1: 655, 2: 210}
    assert mapper.mapping[12] == 2


@pytest.mark.unit
def test_mapper_without_tracker_ids_uses_vertical_order() -> None:
    mapper = TrackIdMapper()
    boxes = mapper.update([(None, 0.9, [0, 0, 10, 100]), (None, 0.9, [0, 0, 10, 600])])
    assert {b.track_id: b.y1 for b in boxes} == {1: 600, 2: 100}
