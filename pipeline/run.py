import argparse
import importlib
from pathlib import Path
from typing import cast

from loguru import logger

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader
from utils import read_yaml


def load_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _output_path(config: dict, key: str) -> Path | None:
    """``config["outputs"][key]["path"]`` (e.g. ``video``, ``heatmaps``), ``None`` when unset."""
    outputs = config.get("outputs") or {}
    section = outputs.get(key) if isinstance(outputs, dict) else None
    path = section.get("path") if isinstance(section, dict) else None
    if isinstance(path, str) and path.strip():
        return Path(path.strip())
    return None


def _render_tennis_outputs(config: dict, video_reader: VideoReader, analysis: Analysis) -> None:
    """Optional visual outputs: overlay video and player heatmaps (tennis only)."""
    from pipeline.objects.tennis.tennis_analysis import TennisAnalysis

    if not isinstance(analysis, TennisAnalysis):
        return
    overlay_path = _output_path(config, "video")
    if overlay_path:
        from pipeline.render.tennis_overlay import render_overlay_video

        render_overlay_video(video_reader, analysis, overlay_path)
    heatmaps_dir = _output_path(config, "heatmaps")
    if heatmaps_dir:
        from pipeline.render.tennis_heatmap import render_heatmaps

        render_heatmaps(analysis, heatmaps_dir)


def run_pipeline(
    sport: str,
    video_path: str,
    config: dict,
    test: bool = False,
) -> dict | None:
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

    if not test and sport == "tennis":
        try:
            _render_tennis_outputs(config, video_reader, analysis)
        except Exception as e:
            logger.exception(f"Error writing the visual outputs: {e}")
            raise

    video_reader.release()

    if test:
        return analysis.to_dict()

    analysis.save_json(config["outputs"].get("path", "outputs/analysis.json"))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to the video")
    parser.add_argument("--config", type=str, required=True, help="path to the config")
    parser.add_argument("--sport", type=str, required=True, help="sport to analyze")
    args = parser.parse_args()

    run_pipeline(args.sport, args.video_path, read_yaml(args.config))
