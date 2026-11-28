"""Unit tests for the ball annotation model (tools/ball_annotator/ball_annotations.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tools.ball_annotator.ball_annotations import CSV_COLUMNS, EVENTS, BallAnnotations

from tests.evaluation.ball_evaluation import match_balls


def _annotations() -> BallAnnotations:
    ann = BallAnnotations()
    ann.set_position(0, 706, 480)
    ann.set_position(1, 704.5, 484)
    ann.toggle_event(1, "serve")
    ann.toggle_event(3, "bounce")  # event without a visible ball
    return ann


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "clip.csv"
    _annotations().save_csv(path, frame_count=5)

    loaded = BallAnnotations.load_csv(path)
    assert loaded == _annotations()


def test_csv_has_one_row_per_frame_and_empty_cells(tmp_path: Path) -> None:
    path = tmp_path / "clip.csv"
    _annotations().save_csv(path, frame_count=5)

    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_COLUMNS)
    assert lines[1:] == [
        "0,706.0,480.0,",
        "1,704.5,484.0,serve",
        "2,,,",
        "3,,,bounce",
        "4,,,",
    ]


def test_frames_annotated_after_frame_count_are_kept() -> None:
    ann = BallAnnotations()
    ann.toggle_event(7, "hit")
    assert len(ann.to_dataframe(frame_count=5)) == 8


def test_legacy_csv_without_event_column(tmp_path: Path) -> None:
    path = tmp_path / "clip.csv"
    path.write_text("frame,x,y\n0,706.0,480.0\n1,,\n")

    ann = BallAnnotations.load_csv(path)
    assert ann.positions == {0: (706.0, 480.0)}
    assert ann.events == {}


def test_unknown_event_in_csv_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "clip.csv"
    path.write_text("frame,x,y,event\n0,1,2,smash\n")

    with pytest.raises(ValueError, match="frame 0"):
        BallAnnotations.load_csv(path)


def test_event_case_and_spaces_are_normalized() -> None:
    df = pd.DataFrame({"frame": [0], "x": [1.0], "y": [2.0], "event": [" Bounce "]})
    assert BallAnnotations.from_dataframe(df).events == {0: "bounce"}


def test_toggle_event_sets_replaces_and_removes() -> None:
    ann = BallAnnotations()
    assert ann.toggle_event(4, "hit") == "hit"
    assert ann.toggle_event(4, "bounce") == "bounce"  # another key replaces
    assert ann.toggle_event(4, "bounce") is None  # same key removes
    assert ann.events == {}


def test_toggle_unknown_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown event"):
        BallAnnotations().toggle_event(0, "net")


def test_delete_position_and_clear_event() -> None:
    ann = _annotations()
    assert ann.delete_position(0) is True
    assert ann.delete_position(0) is False
    assert ann.clear_event(1) is True
    assert ann.clear_event(1) is False
    assert 1 in ann.positions  # clearing the event keeps the ball


def test_event_navigation() -> None:
    ann = BallAnnotations()
    for frame, event in [(10, "serve"), (25, "bounce"), (40, "hit")]:
        ann.toggle_event(frame, event)

    assert ann.next_event(10) == 25
    assert ann.next_event(0) == 10
    assert ann.next_event(40) is None
    assert ann.previous_event(25) == 10
    assert ann.previous_event(10) is None


def test_event_counts_in_events_order() -> None:
    ann = BallAnnotations()
    for frame, event in [(1, "hit"), (2, "bounce"), (3, "hit")]:
        ann.toggle_event(frame, event)
    assert ann.event_counts() == {"serve": 0, "hit": 2, "bounce": 1}
    assert tuple(ann.event_counts()) == EVENTS


def test_ball_evaluator_reads_the_new_csv(tmp_path: Path) -> None:
    """The event column does not change the ball metrics."""
    path = tmp_path / "clip.csv"
    _annotations().save_csv(path, frame_count=5)
    frames = [
        {"id": 0, "ball": {"position": [706, 481]}},  # TP
        {"id": 1},  # FN (annotated, not predicted)
        {"id": 2, "ball": {"position": [10, 10]}},  # FP2 (no ball annotated)
        {"id": 3},  # TN (event but no ball)
        {"id": 4},  # TN
    ]

    counts = match_balls(frames, pd.read_csv(path), min_dist=10)
    assert (counts.tp, counts.fn, counts.fp1, counts.fp2, counts.tn) == (1, 1, 0, 1, 2)
