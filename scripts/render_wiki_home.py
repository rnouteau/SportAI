#!/usr/bin/env python3
"""Render the GitHub Wiki Home.md from NRT report.json files (+ F1 history)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

SUITE_TENNIS_BALL = "tennis_ball_tracking"
SUITE_TENNIS_PLAYER = "tennis_player_detection"
HISTORY_LIMIT = 120
F1_METRIC = "F1-Score"

SUITE_TITLES = {
    SUITE_TENNIS_BALL: "Tennis ball tracking",
    SUITE_TENNIS_PLAYER: "Tennis player detection",
}


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return (
        "\n".join([line, sep, body])
        if rows
        else "\n".join([line, sep, "| " + " | ".join("—" for _ in headers) + " |"])
    )


def load_history(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("runs") or [])
    if isinstance(data, list):
        return data
    return []


def save_history(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def extract_total_f1(report: dict) -> float | None:
    videos = (report.get("current") or {}).get("videos") or {}
    total = videos.get("TOTAL") or {}
    value = total.get(F1_METRIC)
    return float(value) if value is not None else None


def append_run(
    history: list[dict],
    *,
    suite: str,
    date: str,
    f1: float | None,
    commit: str,
    run_url: str,
    conclusion: str,
    limit: int = HISTORY_LIMIT,
) -> list[dict]:
    point = {
        "suite": suite,
        "date": date,
        "f1": f1,
        "commit": commit,
        "run_url": run_url,
        "conclusion": conclusion,
    }
    day = date[:10]
    short = commit[:7] if commit else ""

    replaced = False
    out: list[dict] = []
    for item in history:
        same_run = run_url and item.get("run_url") == run_url and item.get("suite") == suite
        same_commit_day = (
            short
            and (item.get("commit") or "")[:7] == short
            and str(item.get("date", ""))[:10] == day
            and item.get("suite") == suite
        )
        if same_run or same_commit_day:
            out.append(point)
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(point)

    out.sort(key=lambda item: (item.get("date") or "", item.get("run_url") or ""))
    if len(out) > limit:
        out = out[-limit:]
    return out


def _status_label(conclusion: str) -> str:
    lower = (conclusion or "").lower()
    if lower in {"success", "pass", "passed"}:
        return "PASS"
    if lower in {"failure", "fail", "failed"}:
        return "FAIL"
    return (conclusion or "unknown").upper()


def mermaid_f1_chart(history: list[dict], suite: str) -> str:
    points = [item for item in history if item.get("suite") == suite and item.get("f1") is not None]
    if not points:
        return "_No F1 history yet for this suite._"

    labels = [str(item["date"])[:10] for item in points]
    values = [round(float(item["f1"]), 4) for item in points]
    y_min = max(0.0, min(values) - 0.05)
    y_max = min(1.0, max(values) + 0.05)
    if y_max <= y_min:
        y_max = y_min + 0.01

    label_list = ", ".join(f'"{label}"' for label in labels)
    value_list = ", ".join(f"{v:.4f}" for v in values)
    title = SUITE_TITLES.get(suite, suite)
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{title} F1-Score"',
            f"    x-axis [{label_list}]",
            f'    y-axis "F1" {y_min:.2f} --> {y_max:.2f}',
            f"    line [{value_list}]",
            "```",
        ]
    )


def history_table(history: list[dict], suite: str) -> str:
    rows = []
    for item in reversed([h for h in history if h.get("suite") == suite]):
        commit = item.get("commit") or ""
        short = commit[:7] if commit else "—"
        run_url = item.get("run_url") or ""
        commit_cell = f"[`{short}`]({run_url})" if run_url and short != "—" else f"`{short}`"
        rows.append(
            [
                str(item.get("date", ""))[:10] or "—",
                _fmt(item.get("f1")),
                _status_label(str(item.get("conclusion") or "")),
                commit_cell,
            ]
        )
    return _md_table(["Date", "F1", "Status", "Commit"], rows)


def build_metric_rows(
    current: dict, baseline: dict | None, tolerance: float
) -> tuple[list[list[str]], bool]:
    rows: list[list[str]] = []
    all_ok = True
    baseline_videos = (baseline or {}).get("videos", {})
    for video, metrics in current.get("videos", {}).items():
        expected_metrics = baseline_videos.get(video, {})
        for metric, value in metrics.items():
            expected = expected_metrics.get(metric)
            if baseline is None:
                status = "new baseline"
                delta = None
                all_ok = False
            elif expected is None:
                status = "missing from baseline"
                delta = None
                all_ok = False
            else:
                delta = value - expected
                status = "OK" if abs(delta) <= tolerance else "DIFF"
                if status != "OK":
                    all_ok = False
            rows.append(
                [
                    video,
                    metric,
                    _fmt(value),
                    _fmt(expected),
                    _fmt(delta),
                    status,
                ]
            )
    return rows, all_ok and baseline is not None


def build_count_rows(results: list[dict]) -> list[list[str]]:
    rows = []
    for item in results:
        rows.append(
            [
                str(item.get("Video", "")),
                str(item.get("Total", "")),
                str(item.get("TP", "")),
                str(item.get("FP", "")),
                str(item.get("TN", "")),
                str(item.get("FN", "")),
            ]
        )
    return rows


def render_suite_section(report: dict, *, suite: str, tolerance: float) -> tuple[list[str], bool]:
    title = SUITE_TITLES.get(suite, suite)
    current = report["current"]
    baseline = report.get("baseline")
    results = report.get("results") or []
    context = current.get("context") or {}
    metric_rows, metrics_ok = build_metric_rows(current, baseline, tolerance)

    lines = [
        f"## {title}",
        "",
        "### Context",
        "",
        _md_table(
            ["Key", "Value"],
            [[key, f"`{value}`"] for key, value in context.items()] or [["—", "—"]],
        ),
        "",
        "### Metrics vs baseline",
        "",
        _md_table(
            ["Video", "Metric", "Current", "Baseline", "Delta", "Status"],
            metric_rows,
        ),
        "",
        "### Counts",
        "",
        _md_table(
            ["Video", "Total", "TP", "FP", "TN", "FN"],
            build_count_rows(results),
        ),
        "",
    ]
    return lines, metrics_ok


def render_home(
    reports: dict[str, dict],
    history: list[dict],
    *,
    commit: str,
    run_url: str,
    workflow: str,
    conclusion: str,
    tolerance: float,
    generated_at: str | None = None,
) -> str:
    """Render Home.md for one or more suite reports (suite id -> report dict)."""
    suite_oks: list[bool] = []
    section_blocks: list[str] = []
    for suite, report in reports.items():
        lines, ok = render_suite_section(report, suite=suite, tolerance=tolerance)
        suite_oks.append(ok)
        section_blocks.extend(lines)

    metrics_ok = all(suite_oks) if suite_oks else False
    if conclusion.lower() in {"success", "pass", "passed"}:
        status = "PASS" if metrics_ok else "PASS (see metric notes)"
    elif conclusion.lower() in {"failure", "fail", "failed"}:
        status = "FAIL"
    else:
        status = conclusion.upper()

    generated_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    short_sha = commit[:7] if commit else "—"

    lines = [
        "# SportAI — Model test results",
        "",
        "Auto-updated from NRT runs on `main` (CI + nightly). Pull request NRT runs do **not** update this page.",
        "",
        "## Latest run",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["Status", status],
                ["Generated", generated_at],
                ["Workflow", workflow or "—"],
                ["Commit", f"`{short_sha}`" if commit else "—"],
                ["Actions run", f"[open]({run_url})" if run_url else "—"],
                ["Tolerance", _fmt(tolerance)],
            ],
        ),
        "",
        "## F1 over time",
        "",
        "TOTAL `F1-Score` for each NRT publish on `main` (kept in wiki `nrt-history.json`).",
        "",
    ]

    for suite in reports:
        title = SUITE_TITLES.get(suite, suite)
        lines.extend(
            [
                f"### {title}",
                "",
                mermaid_f1_chart(history, suite),
                "",
                history_table(history, suite),
                "",
            ]
        )

    lines.extend(section_blocks)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Ball NRT report.json")
    parser.add_argument(
        "--report-player",
        type=Path,
        default=None,
        help="Optional player NRT report.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to write Home.md")
    parser.add_argument("--history-in", type=Path, default=None, help="Existing nrt-history.json")
    parser.add_argument(
        "--history-out", type=Path, default=None, help="Path to write updated history"
    )
    parser.add_argument("--commit", default="", help="Git commit SHA")
    parser.add_argument("--run-url", default="", help="GitHub Actions run URL")
    parser.add_argument("--workflow", default="", help="Workflow name")
    parser.add_argument("--conclusion", default="unknown", help="Job conclusion (success/failure)")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Baseline metric tolerance")
    parser.add_argument(
        "--suite",
        default=SUITE_TENNIS_BALL,
        help="Suite id for --report (default: tennis_ball_tracking)",
    )
    parser.add_argument(
        "--suite-player",
        default=SUITE_TENNIS_PLAYER,
        help="Suite id for --report-player",
    )
    args = parser.parse_args()

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    history = load_history(args.history_in)

    reports: dict[str, dict] = {}
    ball_report = json.loads(args.report.read_text(encoding="utf-8"))
    reports[args.suite] = ball_report
    history = append_run(
        history,
        suite=args.suite,
        date=generated_at,
        f1=extract_total_f1(ball_report),
        commit=args.commit,
        run_url=args.run_url,
        conclusion=args.conclusion,
    )

    if args.report_player is not None and args.report_player.exists():
        player_report = json.loads(args.report_player.read_text(encoding="utf-8"))
        reports[args.suite_player] = player_report
        history = append_run(
            history,
            suite=args.suite_player,
            date=generated_at,
            f1=extract_total_f1(player_report),
            commit=args.commit,
            run_url=args.run_url,
            conclusion=args.conclusion,
        )

    markdown = render_home(
        reports,
        history,
        commit=args.commit,
        run_url=args.run_url,
        workflow=args.workflow,
        conclusion=args.conclusion,
        tolerance=args.tolerance,
        generated_at=generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")

    history_out = args.history_out or args.history_in
    if history_out is not None:
        save_history(history_out, history)
        print(f"Wrote {history_out}")


if __name__ == "__main__":
    main()
