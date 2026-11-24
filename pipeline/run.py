import argparse
import importlib
import re
from pathlib import Path
from typing import cast

from loguru import logger

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader
from utils import read_yaml

# Names of the files of one analysis, inside its folder (read by the API too).
STATS_FILE = "stats.json"
OVERLAY_FILE = "overlay.mp4"
HEATMAPS_DIR = "heatmaps"


def load_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def analysis_id(video_path: str | Path) -> str:
    """Folder name / URL id of an analysis: the video name in lower case, every run of
    characters other than ``a-z 0-9 _ -`` replaced by ``-``.

    ``Fonseca vs Djokovic ｜ Full match [n2zjuuvgkGY].mp4`` → ``fonseca-vs-djokovic-full-match-n2zjuuvgkgy``
    """
    slug = re.sub(r"[^a-z0-9_-]+", "-", Path(video_path).stem.lower()).strip("-")
    return slug or "analysis"


def analysis_dir(config: dict, sport: str, video_path: str | Path, output_dir: str | None) -> Path:
    """``<output_dir or outputs.dir>/<analysis id>``: everything a run writes goes there."""
    outputs = config.get("outputs") or {}
    root = output_dir or outputs.get("dir") or f"outputs/{sport}"
    return Path(root) / analysis_id(video_path)


def _render_tennis_outputs(
    config: dict, video_reader: VideoReader, analysis: Analysis, out_dir: Path
) -> None:
    """Optional visual outputs (``outputs.video`` / ``outputs.heatmaps``), tennis only."""
    from pipeline.objects.tennis.tennis_analysis import TennisAnalysis

    if not isinstance(analysis, TennisAnalysis):
        return
    outputs = config.get("outputs") or {}
    if outputs.get("video"):
        from pipeline.render.tennis_overlay import render_overlay_video

        render_overlay_video(video_reader, analysis, out_dir / OVERLAY_FILE)
    if outputs.get("heatmaps"):
        from pipeline.render.tennis_heatmap import render_heatmaps

        render_heatmaps(analysis, out_dir / HEATMAPS_DIR)


def run_pipeline(
    sport: str,
    video_path: str,
    config: dict,
    test: bool = False,
    output_dir: str | None = None,
) -> dict | None:
    """Run the models then the stats on a video.

    ``test=True``: return the analysis as a dict, write nothing. Otherwise write
    ``stats.json`` (+ ``overlay.mp4`` / ``heatmaps/`` when enabled in ``outputs``) into
    ``<output_dir or outputs.dir>/<analysis id>/`` and return ``None``.
    """
    logger.info("Running pipeline")
    if not video_path:
        logger.error("Video path is empty")
        return None

    analysis_path = config.get("analysis")
    if not analysis_path or not isinstance(analysis_path, str):
        raise ValueError(
            "config must define 'analysis' as a fully-qualified class path "
            "(e.g. pipeline.objects.tennis.TennisAnalysis)"
        )

    video_reader = VideoReader(video_path)
    analysis_cls = cast(type[Analysis], load_class(analysis_path))
    analysis = analysis_cls.from_video(video_reader, sport)

    for model_cfg in config.get("models", []):
        model_name = model_cfg["name"]
        try:
            model = load_class(model_name)(model_cfg.get("config", {}))
            model.run(video_reader, analysis)
        except Exception as e:
            logger.exception(f"Error occurred while running model {model_name}: {e}")
            raise

    for stat_cfg in config.get("stats", []):
        stat_name = stat_cfg["name"]
        try:
            stat = load_class(stat_name)(stat_cfg.get("config", {}))
            stat.run(video_reader, analysis)
        except Exception as e:
            logger.exception(f"Error occurred while running stat {stat_name}: {e}")
            raise

    if test:
        video_reader.release()
        return analysis.to_dict()

    out_dir = analysis_dir(config, sport, video_path, output_dir)
    if sport == "tennis":
        try:
            _render_tennis_outputs(config, video_reader, analysis, out_dir)
        except Exception as e:
            logger.exception(f"Error writing the visual outputs: {e}")
            raise
    video_reader.release()

    analysis.save_json(out_dir / STATS_FILE)
    logger.info(f"Analysis written to {out_dir}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to the video")
    parser.add_argument("--config", type=str, required=True, help="path to the config")
    parser.add_argument("--sport", type=str, required=True, help="sport to analyze")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="root folder (default: outputs.dir)"
    )
    args = parser.parse_args()

    run_pipeline(args.sport, args.video_path, read_yaml(args.config), output_dir=args.output_dir)
