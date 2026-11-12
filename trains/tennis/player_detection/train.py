import argparse

from loguru import logger
from ultralytics import YOLO

from utils import read_yaml


def train(config: dict) -> None:
    logger.info("Running training")
    model = YOLO()
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the config")
    args = parser.parse_args()
    train(read_yaml(args.config))
