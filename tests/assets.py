
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from utils.paths import get_root


def resolve_dataset_dir(cfg) -> Path:
    local = get_root() / cfg["local_dir"]
    if local.exists():
        return local
    root = snapshot_download(cfg["repo_id"], repo_type="dataset", revision=cfg["revision"],
                             allow_patterns=[f"{cfg['subset']}/*"])
    return Path(root) / cfg["subset"]

def resolve_weights(model_cfg, weights_cfg) -> str:
    local = get_root() / model_cfg["weight"]
    return str(local) if local.exists() else hf_hub_download(weights_cfg["repo_id"], weights_cfg["filename"])
