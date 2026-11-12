from pathlib import Path

import yaml


def read_yaml(file: str | Path) -> dict:
    with open(file, "r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)
    return config
