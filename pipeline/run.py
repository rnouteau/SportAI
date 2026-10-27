from utils import read_yaml
import argparse
from loguru import logger



def run_pipepline(config):
    logger.info("Running pipeline")




if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True,help="path to the config")
    args = parser.parse_args()

    config_path = args.config
    config = read_yaml(config_path)
    run_pipepline(config)

