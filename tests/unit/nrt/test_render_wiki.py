"""Unit tests for the wiki renderer (scripts/render_wiki_home.py)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.render_wiki_home import (
    RunInfo,
    append_run,
    fmt,
    load_history,
    load_reports,
    md_table,
    metric_status,
    render_wiki,
    sparkline,
    suite_status,
)
from tests.nrt.suites import SUITES_BY_ID

PLAYERS = SUITES_BY_ID["tennis_players"]
BALL = SUITES_BY_ID["tennis_ball_detection"]


def _run(**kwargs: object) -> RunInfo:
    defaults: dict = {
        "commit": "0123456789abcdef",
        "run_url": "https://github.com/o/r/actions/runs/1",
        "workflow": "NRT Nightly",
        "conclusion": "success",
        "generated_at": "2026-11-21 02:00 UTC",
        "repo_url": "https://github.com/o/r",
    }
    return RunInfo(**{**defaults, **kwargs})


def _report(videos: dict, baseline: dict | None = None, results: list | None = None) -> dict:
    current = {"context": {"device": "cpu", "weights": "m.pt"}, "videos": videos}
    base = {"context": current["context"], "videos": baseline} if baseline is not None else None
    return {"current": current, "baseline": base, "results": results or []}


def test_fmt_counts_and_ratios() -> None:
    assert fmt(0.5) == "0.5000"
    assert fmt(1.0) == "1.0000"
    assert fmt(2.0, as_int=True) == "2"
    assert fmt(7) == "7"
    assert fmt(None) == "—"
    assert fmt(float("nan")) == "—"


def test_md_table_escapes_pipes() -> None:
    table = md_table(["Video"], [["a | b"]])
    assert "a \\| b" in table


def test_sparkline() -> None:
    assert sparkline([]) == "—"
    assert sparkline([0.5, 0.5]) == "▄▄"
    assert sparkline([0.0, 1.0]) == "▁█"


def test_metric_status_respects_direction_and_tolerance() -> None:
    assert metric_status(PLAYERS, "IDF1", 0.903, 0.9, 0.005) == "same"
    assert metric_status(PLAYERS, "IDF1", 0.95, 0.9, 0.005) == "better"
    assert metric_status(PLAYERS, "ID switches", 3, 1, 0.005) == "worse"
    assert metric_status(PLAYERS, "ID switches", 0, 1, 0.005) == "better"
    assert metric_status(PLAYERS, "IDF1", 0.9, None, 0.005) == "new"


def test_suite_status_is_the_worst_video_metric() -> None:
    base = {"a.mp4": {"IDF1": 0.9, "ID switches": 0.0}, "TOTAL": {"IDF1": 0.9, "ID switches": 0.0}}
    cur = {"a.mp4": {"IDF1": 0.95, "ID switches": 2.0}, "TOTAL": {"IDF1": 0.95, "ID switches": 0.0}}
    assert suite_status(PLAYERS, _report(cur, base), 0.005) == "worse"
    assert suite_status(PLAYERS, _report(base, base), 0.005) == "same"
    assert suite_status(PLAYERS, _report(cur, None), 0.005) == "new"


def test_legacy_history_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "nrt-history.json"
    runs = [
        {"suite": "tennis_ball_tracking", "date": "2026-11-01 02:00:00 UTC", "f1": 0.8},
        {"suite": "tennis_player_tracking", "date": "2026-11-02", "metrics": {"IDF1": 0.9}},
        {"suite": "tennis_player_detection", "date": "2026-11-02", "metrics": {"F1-Score": 0.9}},
    ]
    path.write_text(json.dumps({"runs": runs}))
    history = load_history(path)
    assert history == [
        {
            "suite": "tennis_ball_detection",
            "date": "2026-11-01 02:00:00 UTC",
            "metrics": {"F1-Score": 0.8},
        },
        {"suite": "tennis_players", "date": "2026-11-02", "metrics": {"IDF1": 0.9}},
        # merged into tennis_players: kept in the history file, no longer rendered
        {"suite": "tennis_player_detection", "date": "2026-11-02", "metrics": {"F1-Score": 0.9}},
    ]


def test_append_run_replaces_rerun_of_same_commit_and_day() -> None:
    point = {"suite": "s", "date": "2026-11-21 02:00 UTC", "commit": "abc1234ff", "run_url": "u1"}
    history = append_run([], point)
    history = append_run(history, {**point, "date": "2026-11-21 09:00 UTC", "run_url": "u2"})
    history = append_run(history, {**point, "suite": "other"})
    # sorted by date: "other" (02:00) then the re-run of "s" (09:00), which replaced u1
    assert [(r["suite"], r["run_url"]) for r in history] == [("other", "u1"), ("s", "u2")]
    assert len(append_run(history, {**point, "date": "2026-11-22 02:00 UTC", "run_url": "u3"})) == 3


def test_load_reports_only_known_suites(tmp_path: Path) -> None:
    (tmp_path / "tennis_players").mkdir()
    (tmp_path / "tennis_players" / "report.json").write_text(json.dumps(_report({})))
    (tmp_path / "unknown_suite").mkdir()
    (tmp_path / "unknown_suite" / "report.json").write_text("{}")
    assert list(load_reports(tmp_path)) == ["tennis_players"]


def test_render_wiki_pages_and_history() -> None:
    videos = {
        "clip | 1.mp4": {"IDF1": 0.8, "ID switches": 2.0},
        "TOTAL": {"IDF1": 0.8, "ID switches": 2.0},
    }
    results = [{"Video": "clip | 1.mp4", "IDF1": 0.8, "ID switches": 2.0, "FP": 3, "FN": 4}]
    reports = {"tennis_players": _report(videos, videos, results)}
    old = [
        {
            "suite": "tennis_court_detection",
            "date": "2026-11-20 02:00 UTC",
            "metrics": {"F1-Score": 0.9},
        }
    ]

    pages, history = render_wiki(reports, old, _run())

    assert set(pages) == {
        "Home.md",
        "_Sidebar.md",
        "Tennis-ball-detection.md",
        "Tennis-players.md",
        "Tennis-court-detection.md",
    }
    home = pages["Home.md"]
    assert "[Tennis players](Tennis-players) | IDF1 | 0.8000 | 0.8000 | 0" in home
    assert "✅ same as baseline" in home
    assert "| [Tennis court detection](Tennis-court-detection) | F1-Score | 0.9000 |" in home
    assert "⚪ not run" in home
    tracking = pages["Tennis-players.md"]
    assert "ID switches (lower is better) | 2 | 2 | 0 |" in tracking
    assert "clip \\| 1.mp4 | 0.8000 | 2 | 3 | 4 |" in tracking
    assert "[`0123456`](https://github.com/o/r/commit/0123456789abcdef)" in tracking
    assert history[-1]["suite"] == "tennis_players"
    assert history[-1]["metrics"] == {"IDF1": 0.8, "ID switches": 2.0}
    assert "[Tennis court detection](Tennis-court-detection)" in pages["_Sidebar.md"]


def test_chart_after_two_runs() -> None:
    videos = {"TOTAL": {"F1-Score": 0.9}}
    reports = {"tennis_ball_detection": _report(videos, videos)}
    pages, history = render_wiki(reports, [], _run())
    assert "appears after two runs" in pages["Tennis-ball-detection.md"]
    pages, _ = render_wiki(
        reports,
        history,
        _run(generated_at="2026-11-22 02:00 UTC", commit="fedcba9876543210", run_url="u2"),
    )
    assert "xychart-beta" in pages["Tennis-ball-detection.md"]
    assert BALL.main_metric == "F1-Score"
