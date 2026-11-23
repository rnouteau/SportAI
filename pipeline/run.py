import argparse
import importlib
from typing import cast

from loguru import logger

from pipeline.objects.analysis import Analysis
from pipeline.video_reader import VideoReader
from utils import read_yaml


def load_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _overlay_video_path(config: dict) -> str | None:
    outputs = config.get("outputs") or {}
    if not isinstance(outputs, dict):
        return None
    video = outputs.get("video")
    if not isinstance(video, dict):
        return None
    path = video.get("path")
    if isinstance(path, str) and path.strip():
        return path.strip()
    return None


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

    if not test:
        overlay_path = _overlay_video_path(config)
        if overlay_path and sport == "tennis":
            from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
            from pipeline.render.tennis_overlay import render_overlay_video

            if isinstance(analysis, TennisAnalysis):
                try:
                    render_overlay_video(video_reader, analysis, overlay_path)
                except Exception as e:
                    logger.exception(f"Error writing overlay video: {e}")
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
