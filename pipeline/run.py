from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.video_reader import VideoReader
from pipeline.models.model import Model
import importlib
from utils import read_yaml
import argparse
from loguru import logger


def load_class(path : str):
    module_name,class_name = path.rsplit(".",1)
    return getattr(importlib.import_module(module_name),class_name)

def run_pipeline(sport,video_path,config):
    logger.info("Running pipeline")
    if not video_path:
        logger.error("Video path is empty")
        return

    video_reader = VideoReader(video_path)

    if sport == "tennis":
        analysis = TennisAnalysis.from_video(video_reader,sport)
    else:
        logger.error(f"Sport {sport} not supported")
        return
    
    for model_cfg in config.get("models", []):
        model_name = model_cfg["name"]
        try:
            script = load_class(model_name)(model_cfg.get("config", {}))
            script.run(video_reader, analysis)
        except Exception as e:
            logger.exception(f"Error occurred while running model {model_name}: {e}")

    # for stat in config["stats"]:
    #     try:
    #         stat_name = stat["name"]
    #         stat_options = stat["options"]
    #         logger.info(f"Running stat: {stat_name} with options: {stat_options}")
    #         stat.setup(stat_options)
    #         stat.run(stat_options)
    #     except Exception as e:
    #         logger.error(f"Error occurred while running stat {stat_name}: {e}")
    print(config["outputs"])
    analysis.save_json(config["outputs"].get("dir","outputs/"))





if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True,help="path to the video")
    parser.add_argument("--config",type=str,required=True,help="path to the config")
    parser.add_argument("--sport",type=str,required=True,help="sport to analyze")
    args = parser.parse_args()

    config_path = args.config
    sport = args.sport
    video_path = args.video_path
    config = read_yaml(config_path)
   
    run_pipeline(sport,video_path,config)

