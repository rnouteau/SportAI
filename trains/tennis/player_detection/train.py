from utils import read_yaml
import argparse
from loguru import logger
from ultralytics import YOLO


def train(config):
    logger.info("Running training")

    model = YOLO()

    model.train()







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True,help="path to the config")
    args = parser.parse_args()
    
    config_path = args.config
    config = read_yaml(config_path)
    train(config)

