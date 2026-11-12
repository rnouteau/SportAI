import argparse
import importlib

from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.video_reader import VideoReader
from utils import read_yaml


def load_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


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

    video_reader = VideoReader(video_path)

    if sport == "tennis":
        analysis = TennisAnalysis.from_video(video_reader, sport)
    else:
        logger.error(f"Sport {sport} not supported")
        return None

    for model_cfg in config.get("models", []):
        model_name = model_cfg["name"]
        try:
            model = load_class(model_name)(model_cfg.get("config", {}))
            model.run(video_reader, analysis)
        except Exception as e:
            logger.exception(f"Error occurred while running model {model_name}: {e}")
            raise

    video_reader.release()

    if test:
        return analysis.data

    analysis.save_json(config["outputs"].get("path", "outputs/analysis.json"))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to the video")
    parser.add_argument("--config", type=str, required=True, help="path to the config")
    parser.add_argument("--sport", type=str, required=True, help="sport to analyze")
    args = parser.parse_args()

    run_pipeline(args.sport, args.video_path, read_yaml(args.config))
