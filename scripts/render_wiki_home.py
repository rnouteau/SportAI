#!/usr/bin/env python3
"""Render the GitHub Wiki from the NRT ``report.json`` files.

Pages written to ``--wiki-dir`` (the checked-out ``<repo>.wiki`` repository):

- ``Home.md``           latest run + one summary row per suite
- ``<Suite-page>.md``   one page per suite: context, metrics vs baseline, per-video
                        results, history chart
- ``_Sidebar.md``       navigation
- ``nrt-history.json``  every TOTAL metric of every run (read, updated, written back)

Each suite is found at ``<reports-dir>/<suite id>/report.json`` (written by
``tests/nrt/harness.py``); the suites are listed in ``tests/nrt/suites.py``. Suites
without a report are kept on the wiki with their history but marked as not run.

    python scripts/render_wiki_home.py --reports-dir outputs/tests --wiki-dir wiki-repo \
        --commit "$SHA" --run-url "$URL" --workflow "NRT Nightly" --conclusion success
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# The suite registry lives with the NRT tests; this script runs from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.nrt.suites import LEGACY_SUITE_IDS, SUITES, Suite  # noqa: E402

HISTORY_FILE = "nrt-history.json"
HISTORY_VERSION = 2
HISTORY_LIMIT = 500  # runs kept, all suites together
CHART_POINTS = 30
TREND_POINTS = 10
TOTAL = "TOTAL"
DEFAULT_TOLERANCE = 0.005


STATUS_LABELS = {
    "same": "✅ same as baseline",
    "better": "🟢 better",
    "worse": "🔴 worse",
    "new": "🆕 new baseline",
    "missing": "⚪ not run",
}


# --------------------------------------------------------------------------- formatting
def fmt(value: object, *, as_int: bool = False, digits: int = 4) -> str:
    """Metric value: integers for counts, fixed decimals for ratios."""
    if value is None or isinstance(value, bool):
        return "—" if value is None else str(value)
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return "—"
        if as_int or isinstance(value, int):
            return str(round(value))
        return f"{value:.{digits}f}"
    return str(value)


def fmt_delta(delta: float | None, *, as_int: bool = False) -> str:
    if delta is None:
        return "—"
    if abs(delta) < 5e-5:
        return "0"
    return f"{round(delta):+d}" if as_int else f"{delta:+.4f}"


def cell(text: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    if not rows:
        rows = [["—"] * len(headers)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def sparkline(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return "—"
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return blocks[3] * len(values)
    return "".join(blocks[round((v - low) / (high - low) * (len(blocks) - 1))] for v in values)


def page_link(suite: Suite) -> str:
    return f"[{suite.title}]({suite.page})"


def commit_link(commit: str, repo_url: str) -> str:
    if not commit:
        return "—"
    short = f"`{commit[:7]}`"
    return f"[{short}]({repo_url}/commit/{commit})" if repo_url else short


# --------------------------------------------------------------------------- reports
def load_reports(reports_dir: Path) -> dict[str, dict]:
    """``{suite id: report}`` for every known suite that has a report.json."""
    reports: dict[str, dict] = {}
    for suite in SUITES:
        path = reports_dir / suite.id / "report.json"
        if path.exists():
            reports[suite.id] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def current_videos(report: dict) -> dict[str, dict]:
    return (report.get("current") or {}).get("videos") or {}


def baseline_videos(report: dict) -> dict[str, dict]:
    return (report.get("baseline") or {}).get("videos") or {}


def total_metrics(report: dict) -> dict[str, float]:
    videos = current_videos(report)
    if TOTAL in videos:
        return dict(videos[TOTAL])
    return dict(next(iter(videos.values()), {}))


def metric_status(
    suite: Suite, metric: str, value: float | None, expected: float | None, tolerance: float
) -> str:
    if value is None or expected is None:
        return "new"
    delta = value - expected
    if abs(delta) <= tolerance:
        return "same"
    improved = delta < 0 if metric in suite.lower_is_better else delta > 0
    return "better" if improved else "worse"


def suite_status(suite: Suite, report: dict, tolerance: float) -> str:
    """Worst status over every metric of every video."""
    if not report.get("baseline"):
        return "new"
    expected_videos = baseline_videos(report)
    statuses = set()
    for video, metrics in current_videos(report).items():
        expected = expected_videos.get(video, {})
        for metric, value in metrics.items():
            statuses.add(metric_status(suite, metric, value, expected.get(metric), tolerance))
    for status in ("worse", "new", "better"):
        if status in statuses:
            return status
    return "same"


# --------------------------------------------------------------------------- history
def load_history(path: Path) -> list[dict]:
    """Runs from ``nrt-history.json`` (v2, or v1 with only ``f1``, migrated)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") if isinstance(data, dict) else data
    out = []
    for run in runs or []:
        run = dict(run)
        run["suite"] = LEGACY_SUITE_IDS.get(run.get("suite", ""), run.get("suite", ""))
        if "metrics" not in run:
            f1 = run.pop("f1", None)
            run["metrics"] = {"F1-Score": f1} if f1 is not None else {}
        out.append(run)
    return out


def save_history(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": HISTORY_VERSION, "runs": runs}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_run(history: list[dict], point: dict, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Add one run; replaces a run of the same suite with the same Actions run URL,
    or with the same commit on the same day (re-runs)."""
    suite, run_url = point["suite"], point.get("run_url") or ""
    short, day = (point.get("commit") or "")[:7], str(point["date"])[:10]
    out = []
    for run in history:
        same_suite = run.get("suite") == suite
        same_run = bool(run_url) and run.get("run_url") == run_url
        same_commit_day = (
            bool(short)
            and (run.get("commit") or "")[:7] == short
            and str(run.get("date", ""))[:10] == day
        )
        if not (same_suite and (same_run or same_commit_day)):
            out.append(run)
    out.append(point)
    out.sort(key=lambda run: (str(run.get("date") or ""), str(run.get("run_url") or "")))
    return out[-limit:]


def metric_history(history: list[dict], suite: Suite, metric: str) -> list[tuple[str, float]]:
    points = []
    for run in history:
        if run.get("suite") != suite.id:
            continue
        value = (run.get("metrics") or {}).get(metric)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            points.append((str(run.get("date", "")), float(value)))
    return points


def mermaid_chart(suite: Suite, metric: str, points: list[tuple[str, float]]) -> str:
    points = points[-CHART_POINTS:]
    if len(points) < 2:
        return f"_The {metric} chart appears after two runs._"
    labels = ", ".join(f'"{date[5:10]}"' for date, _ in points)
    values = [round(v, 4) for _, v in points]
    pad = max((max(values) - min(values)) * 0.2, 0.01)
    low, high = min(values) - pad, max(values) + pad
    if metric not in suite.lower_is_better and max(values) <= 1.0:
        low, high = max(0.0, low), min(1.0, high)
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{suite.title} — {metric}"',
            f"    x-axis [{labels}]",
            f'    y-axis "{metric}" {low:.3f} --> {high:.3f}',
            f"    line [{', '.join(f'{v:.4f}' for v in values)}]",
            "```",
        ]
    )


# --------------------------------------------------------------------------- pages
@dataclass(frozen=True)
class RunInfo:
    commit: str
    run_url: str
    workflow: str
    conclusion: str
    generated_at: str
    repo_url: str = ""
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def status(self) -> str:
        value = self.conclusion.lower()
        if value in {"success", "pass", "passed"}:
            return "✅ passed"
        if value in {"failure", "fail", "failed"}:
            return "❌ failed"
        return value or "unknown"


def latest_run_line(run: RunInfo) -> str:
    parts = [
        f"**{run.status}**",
        run.generated_at,
        run.workflow or "—",
        f"commit {commit_link(run.commit, run.repo_url)}",
    ]
    if run.run_url:
        parts.append(f"[Actions run]({run.run_url})")
    return " · ".join(parts)


def render_home(reports: dict[str, dict], history: list[dict], run: RunInfo) -> str:
    rows = []
    for suite in SUITES:
        report = reports.get(suite.id)
        trend = sparkline(
            [v for _, v in metric_history(history, suite, suite.main_metric)][-TREND_POINTS:]
        )
        if report is None:
            last = metric_history(history, suite, suite.main_metric)
            rows.append(
                [
                    page_link(suite),
                    suite.main_metric,
                    fmt(last[-1][1]) if last else "—",
                    "—",
                    "—",
                    STATUS_LABELS["missing"],
                    trend,
                ]
            )
            continue
        value = total_metrics(report).get(suite.main_metric)
        expected = baseline_videos(report).get(TOTAL, {}).get(suite.main_metric)
        delta = None if value is None or expected is None else value - expected
        rows.append(
            [
                page_link(suite),
                suite.main_metric,
                fmt(value),
                fmt(expected),
                fmt_delta(delta),
                STATUS_LABELS[suite_status(suite, report, run.tolerance)],
                trend,
            ]
        )

    return "\n".join(
        [
            "# SportAI — model test results",
            "",
            f"Latest run: {latest_run_line(run)}",
            "",
            md_table(["Suite", "Key metric", "Current", "Baseline", "Δ", "Status", "Trend"], rows),
            "",
            "**Status** compares every metric of every video with the committed baseline "
            f"(tolerance ±{run.tolerance}). *Better* or *worse* means the baseline should be "
            "reviewed, then regenerated with `--update-baseline`. "
            f"**Trend**: key metric over the last {TREND_POINTS} runs.",
            "",
            "Updated automatically by the NRT runs on `main` (CI and nightly). "
            "Pull request runs do not update the wiki.",
            "",
        ]
    )


def context_table(report: dict) -> str:
    context = (report.get("current") or {}).get("context") or {}
    rows = [[key.replace("_", " "), f"`{value}`"] for key, value in context.items()]
    return md_table(["Setting", "Value"], rows)


def totals_table(suite: Suite, report: dict, tolerance: float) -> str:
    current = total_metrics(report)
    expected = baseline_videos(report).get(TOTAL, {})
    rows = []
    for metric, value in current.items():
        base = expected.get(metric)
        delta = None if base is None else value - base
        label = f"**{metric}**" if metric == suite.main_metric else metric
        hint = " (lower is better)" if metric in suite.lower_is_better else ""
        as_int = metric in suite.count_metrics
        rows.append(
            [
                label + hint,
                fmt(value, as_int=as_int),
                fmt(base, as_int=as_int),
                fmt_delta(delta, as_int=as_int),
                STATUS_LABELS[metric_status(suite, metric, value, base, tolerance)],
            ]
        )
    return md_table(["Metric", "Current", "Baseline", "Δ", "Status"], rows)


def videos_table(suite: Suite, report: dict) -> tuple[str, int]:
    """Per-video table: regression metrics + counts from the raw results."""
    results = [r for r in report.get("results") or [] if r.get("Video") != TOTAL]
    current = current_videos(report)
    if not results and not current:
        return "_No per-video results._", 0
    metrics = list(next(iter(current.values()), {}).keys())
    counts: list[str] = []
    for row in results:
        for key, value in row.items():
            if key == "Video" or key in metrics or key in counts:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                counts.append(key)
    by_name = {r["Video"]: r for r in results}
    names = [v for v in current if v != TOTAL] or list(by_name)
    rows = []
    for name in names:
        values = {**by_name.get(name, {}), **current.get(name, {})}
        cells = [
            fmt(values.get(m), as_int=m in suite.count_metrics or m in counts)
            for m in metrics + counts
        ]
        rows.append([name, *cells])
    return md_table(["Video", *metrics, *counts], rows), len(rows)


def history_table(history: list[dict], suite: Suite, run: RunInfo, limit: int = 10) -> str:
    runs = [r for r in history if r.get("suite") == suite.id][-limit:]
    rows = []
    for item in reversed(runs):
        value = (item.get("metrics") or {}).get(suite.main_metric)
        link = f"[run]({item['run_url']})" if item.get("run_url") else "—"
        rows.append(
            [
                str(item.get("date", ""))[:16],
                commit_link(item.get("commit") or "", run.repo_url),
                fmt(value, as_int=suite.main_metric in suite.count_metrics),
                item.get("conclusion") or "—",
                link,
            ]
        )
    return md_table(["Date", "Commit", suite.main_metric, "Job", "Link"], rows)


def render_suite_page(suite: Suite, report: dict | None, history: list[dict], run: RunInfo) -> str:
    lines = [f"# {suite.title}", "", suite.description, ""]
    if report is None:
        lines += [
            f"> ⚪ Not run in the latest NRT run ({run.generated_at}). "
            "History below is from earlier runs.",
            "",
        ]
    else:
        status = STATUS_LABELS[suite_status(suite, report, run.tolerance)]
        lines += [
            f"Latest run: {latest_run_line(run)} — **{status}**",
            "",
            "## Results (TOTAL)",
            "",
            totals_table(suite, report, run.tolerance),
            "",
        ]
        table, n_videos = videos_table(suite, report)
        lines += [f"## Per video ({n_videos})", ""]
        if n_videos > 10:
            lines += ["<details><summary>Show the table</summary>", "", table, "", "</details>"]
        else:
            lines.append(table)
        lines += ["", "## Context", "", context_table(report), ""]

    lines += [
        "## History",
        "",
        mermaid_chart(suite, suite.main_metric, metric_history(history, suite, suite.main_metric)),
        "",
        history_table(history, suite, run),
        "",
        "[← Back to Home](Home)",
        "",
    ]
    return "\n".join(lines)


def render_sidebar() -> str:
    lines = ["**SportAI wiki**", "", "- [Home](Home)", "", "**Test suites**", ""]
    lines += [f"- {page_link(suite)}" for suite in SUITES]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main
def render_wiki(
    reports: dict[str, dict], history: list[dict], run: RunInfo
) -> tuple[dict[str, str], list[dict]]:
    """Update the history with this run and return ``({file name: markdown}, history)``."""
    for suite_id, report in reports.items():
        history = append_run(
            history,
            {
                "suite": suite_id,
                "date": run.generated_at,
                "commit": run.commit,
                "run_url": run.run_url,
                "conclusion": run.conclusion,
                "metrics": total_metrics(report),
            },
        )
    pages = {"Home.md": render_home(reports, history, run), "_Sidebar.md": render_sidebar()}
    for suite in SUITES:
        pages[f"{suite.page}.md"] = render_suite_page(suite, reports.get(suite.id), history, run)
    return pages, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GitHub Wiki from NRT reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/tests"))
    parser.add_argument("--wiki-dir", type=Path, required=True, help="Checked-out wiki repo")
    parser.add_argument("--commit", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--repo-url", default="", help="https://github.com/<owner>/<repo>")
    parser.add_argument("--workflow", default="")
    parser.add_argument("--conclusion", default="unknown")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    reports = load_reports(args.reports_dir)
    if not reports:
        raise SystemExit(f"No report.json found under {args.reports_dir}")

    run = RunInfo(
        commit=args.commit,
        run_url=args.run_url,
        workflow=args.workflow,
        conclusion=args.conclusion,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        repo_url=args.repo_url.rstrip("/"),
        tolerance=args.tolerance,
    )
    history_path = args.wiki_dir / HISTORY_FILE
    pages, history = render_wiki(reports, load_history(history_path), run)

    args.wiki_dir.mkdir(parents=True, exist_ok=True)
    for name, markdown in pages.items():
        (args.wiki_dir / name).write_text(markdown, encoding="utf-8")
    save_history(history_path, history)
    print(f"Wrote {len(pages)} pages + {HISTORY_FILE} to {args.wiki_dir} ({', '.join(reports)})")


if __name__ == "__main__":
    main()
