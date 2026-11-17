from pathlib import Path

import yaml

# utils/ -> player_detection/ -> tennis/ -> trains/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as conf_file:
        return yaml.safe_load(conf_file)


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ((base or REPO_ROOT) / candidate).resolve()
