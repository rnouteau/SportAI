from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from utils import read_yaml
import argparse
from loguru import logger



def run_pipepline(config):
    logger.info("Running pipeline")
    sport = config["sport"]
    video_path = config["video_path"]
    if not video_path:
        logger.error("Video path is empty")
        return
    if sport == "tennis":
        analysis = TennisAnalysis(config["analysis"])
    else:
        logger.error(f"Sport {sport} not supported")
        return
    for model in config["models"]:
        try:
            model_name = model["name"]
            model_options = model["options"]
            logger.info(f"Running model: {model_name} with options: {model_options}")
            model.setup(model_options)
            model.run(model_options)
        except Exception as e:
            logger.error(f"Error occurred while running model {model_name}: {e}")

    for stat in config["stats"]:
        try:
            stat_name = stat["name"]
            stat_options = stat["options"]
            logger.info(f"Running stat: {stat_name} with options: {stat_options}")
            stat.setup(stat_options)
            stat.run(stat_options)
        except Exception as e:
            logger.error(f"Error occurred while running stat {stat_name}: {e}")





if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True,help="path to the video")
    parser.add_argument("--config",type=str,required=True,help="path to the config")
    parser.add_argument("--sport",type=str,required=True,help="sport to analyze")
    args = parser.parse_args()

    config_path = args.config
    config = read_yaml(config_path)
    run_pipepline(config)

