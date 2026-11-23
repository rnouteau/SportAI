"""The NRT suites: one registry shared by the NRT tests and the wiki renderer.

Standard library only: ``scripts/render_wiki_home.py`` imports it in the wiki job,
where the ML dependencies are not installed. Evaluators are referenced by name and
imported only when a suite runs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Suite:
    id: str  # folder under outputs/tests, baseline / config file name
    title: str
    page: str  # wiki page name (file name without .md)
    main_metric: str
    description: str
    evaluator: str  # "module:Class"
    config: str  # path from the repository root
    count_columns: tuple[str, ...]  # counts shown in the report next to the metrics
    lower_is_better: frozenset[str] = frozenset()
    count_metrics: frozenset[str] = frozenset()  # regression metrics shown as integers
    min_samples: int = 1  # fewer evaluated videos / images = the dataset is missing

    @property
    def config_path(self) -> Path:
        return ROOT / self.config

    @property
    def report_dir(self) -> Path:
        return ROOT / "outputs" / "tests" / self.id

    def load_evaluator(self) -> type:
        module_name, class_name = self.evaluator.split(":")
        return getattr(importlib.import_module(module_name), class_name)


SUITES: tuple[Suite, ...] = (
    Suite(
        id="tennis_ball_detection",
        title="Tennis ball detection",
        page="Tennis-ball-detection",
        main_metric="F1-Score",
        description="TrackNet ball position per frame, matched to the annotated ball.",
        evaluator="tests.evaluation.ball_evaluation:BallEvaluator",
        config="tests/configs/nrt/tennis/tennis_ball_detection.yaml",
        count_columns=("Total", "TP", "FP1", "FP2", "TN", "FN"),
    ),
    Suite(
        id="tennis_players",
        title="Tennis players",
        page="Tennis-players",
        main_metric="IDF1",
        description=(
            "The two players, from one MOT annotation: boxes (Precision, Recall, Mean IoU), "
            "identities over time (IDF1 = share of the time each player keeps one id, "
            "ID switches) and roles (Role accuracy = P1 really is the near player)."
        ),
        evaluator="tests.evaluation.player_evaluation:PlayerEvaluator",
        config="tests/configs/nrt/tennis/tennis_players.yaml",
        count_columns=("GT", "TP", "FP", "FN", "ID switches"),
        lower_is_better=frozenset({"ID switches"}),
        count_metrics=frozenset({"ID switches"}),
    ),
    Suite(
        id="tennis_court_detection",
        title="Tennis court detection",
        page="Tennis-court-detection",
        main_metric="F1-Score",
        description="The 14 court keypoints, matched to the annotation (PCK, OKS, F1).",
        evaluator="tests.evaluation.court_evaluation:CourtEvaluator",
        config="tests/configs/nrt/tennis/tennis_court_detection.yaml",
        count_columns=("TP", "FP", "FN"),
    ),
)
SUITES_BY_ID = {suite.id: suite for suite in SUITES}

# Suite ids of older versions, mapped to the suite that replaced them (wiki history).
LEGACY_SUITE_IDS = {
    "tennis_ball_tracking": "tennis_ball_detection",
    "tennis_player_tracking": "tennis_players",
}
