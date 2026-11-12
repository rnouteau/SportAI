from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_root() -> Path:
    return ROOT
