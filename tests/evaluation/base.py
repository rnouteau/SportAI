"""Common loop of the evaluators: samples → one result row each → a TOTAL row.

An evaluator only says where its samples are (``samples``), how to score one sample
(``evaluate``) and how to combine them (``total``). Rows are plain dicts
``{"Video": name, <counts>, <metrics>}``: the NRT harness turns them into the
baseline and the report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from pipeline.run import run_pipeline

TOTAL = "TOTAL"


@dataclass
class SampleResult:
    """``row`` goes to the report; ``extra`` keeps what ``total`` needs (counts, accumulators)."""

    row: dict
    extra: Any = field(default=None)


class Evaluator(ABC):
    #: Metric columns printed in the log table (counts are in the report).
    columns: tuple[str, ...] = ()

    def __init__(self, config: dict, directory: Path) -> None:
        self.config = config
        self.directory = Path(directory)
        self.evaluation: dict = config.get("evaluation") or {}
        self.results: list[dict] = []

    @abstractmethod
    def samples(self) -> Iterator[tuple[str, Any]]:
        """``(name, sample)`` pairs; skip (with a warning) samples without annotation."""

    @abstractmethod
    def evaluate(self, sample: Any) -> SampleResult | None:
        """Score one sample, or ``None`` (logged) when it cannot be read."""

    @abstractmethod
    def total(self, results: list[SampleResult]) -> dict:
        """The TOTAL row (without its ``Video`` key)."""

    def run_test(self) -> list[dict]:
        sample_results: list[SampleResult] = []
        for name, sample in self.samples():
            result = self.evaluate(sample)
            if result is None:
                logger.warning(f"{name}: could not be evaluated, skipped")
                continue
            result.row = {"Video": name, **result.row}
            sample_results.append(result)
        self.results = [r.row for r in sample_results]
        if sample_results:
            self.results.append({"Video": TOTAL, **self.total(sample_results)})
        self.print_result()
        return self.results

    def print_result(self) -> None:
        if not self.results:
            logger.warning(f"{type(self).__name__}: no sample evaluated in {self.directory}")
            return
        table = pd.DataFrame(self.results)
        columns = ["Video", *(c for c in self.columns if c in table.columns)]
        logger.info("\n" + table[columns].to_string(index=False, float_format="{:.3f}".format))


class VideoEvaluator(Evaluator):
    """Evaluator whose samples are ``*.mp4`` files run through the whole pipeline."""

    @abstractmethod
    def ground_truth_path(self, video_path: Path) -> Path:
        """Where the annotation of this video is expected."""

    def samples(self) -> Iterator[tuple[str, Any]]:
        for video_path in sorted(self.directory.rglob("*.mp4")):
            gt_path = self.ground_truth_path(video_path)
            if not gt_path.exists():
                logger.warning(f"No annotation for {video_path.name} ({gt_path.name}), skipped")
                continue
            yield video_path.name, (video_path, gt_path)

    def evaluate(self, sample: tuple[Path, Path]) -> SampleResult:
        video_path, gt_path = sample
        analysis = run_pipeline("tennis", str(video_path), self.config, test=True)
        if analysis is None:
            raise RuntimeError(f"The pipeline returned nothing for {video_path.name}")
        return self.evaluate_frames(analysis["frames"], gt_path)

    @abstractmethod
    def evaluate_frames(self, frames: list[dict], gt_path: Path) -> SampleResult:
        """Score the pipeline output (``analysis["frames"]``) against one annotation file."""
