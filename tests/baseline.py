import json
from pathlib import Path


def build_baseline(results: list[dict], context: dict, metrics: list[str]) -> dict:
    return {
        "context": context,
        "videos": {
            row["Video"]: {metric: round(float(row[metric]), 4) for metric in metrics}
            for row in results
        },
    }


def load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def compare_to_baseline(current: dict, baseline: dict, tolerance: float) -> list[str]:
    context_problems = [
        f"context.{key}: baseline={baseline['context'].get(key)!r}, run={value!r}"
        for key, value in current["context"].items()
        if baseline["context"].get(key) != value
    ]
    if context_problems:
        return context_problems + [
            "The run context changed: regenerate the baseline with --update-baseline"
        ]

    problems: list[str] = []
    current_videos = current["videos"]
    baseline_videos = baseline["videos"]

    for video in sorted(baseline_videos.keys() - current_videos.keys()):
        problems.append(f"{video}: video in the baseline but missing from the run")
    for video in sorted(current_videos.keys() - baseline_videos.keys()):
        problems.append(f"{video}: video missing from the baseline")

    for video in sorted(current_videos.keys() & baseline_videos.keys()):
        for metric, value in current_videos[video].items():
            expected = baseline_videos[video].get(metric)
            if expected is None:
                problems.append(f"{video} {metric}: metric missing from the baseline")
            elif abs(value - expected) > tolerance:
                problems.append(
                    f"{video} {metric}: {expected:.4f} -> {value:.4f} ({value - expected:+.4f})"
                )

    return problems
