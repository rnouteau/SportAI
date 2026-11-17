import argparse
import os
import shutil
from pathlib import Path

from loguru import logger
from ultralytics import YOLO

from scripts.prepare_dataset import prepare_dataset
from utils.config import load_config, resolve_path


def _configure_mlflow(mlflow_cfg: dict, run_name: str) -> None:
    if not mlflow_cfg.get("enabled", False):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        os.environ.pop("MLFLOW_RUN", None)
        logger.info("MLflow disabled")
        return

    import mlflow
    from ultralytics import settings

    # Recent MLflow rejects the file store; use SQLite (Windows-safe via as_posix).
    db_path = resolve_path(mlflow_cfg.get("tracking_uri", "mlflow.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path.resolve().as_posix()}"
    experiment = mlflow_cfg.get("experiment_name", "tennis_player_detection")

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    os.environ["MLFLOW_RUN"] = run_name
    settings.update({"mlflow": True})

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
    logger.info("MLflow tracking_uri={} experiment={}", tracking_uri, experiment)


def run(config: dict) -> None:
    paths = config["paths"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    pipeline = config["pipeline"]
    train_cfg = config.get("train", {})
    mlflow_cfg = config.get("mlflow", {})

    videos_dir = resolve_path(paths["videos_dir"])
    dataset_dir = resolve_path(paths["dataset_dir"])
    based_models_dir = resolve_path(paths["based_models_dir"])
    trained_models_dir = resolve_path(paths["trained_models_dir"])
    data_yaml = dataset_dir / "data.yaml"

    trained_models_dir.mkdir(parents=True, exist_ok=True)

    if pipeline.get("prepare_dataset", False):
        logger.info("Preparing dataset from {}", videos_dir)
        data_yaml = prepare_dataset(
            videos_dir=videos_dir,
            dataset_dir=dataset_dir,
            class_names=dataset_cfg.get("class_names", ["player"]),
            annotations_suffix=dataset_cfg.get("annotations_suffix", "_players.csv"),
            image_ext=dataset_cfg.get("image_ext", ".jpg"),
            val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
            split_seed=int(dataset_cfg.get("split_seed", 42)),
            chunk_size=int(dataset_cfg.get("chunk_size", 32)),
        )

    if not pipeline.get("train", False):
        logger.info("Training skipped (pipeline.train=false)")
        return

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing {data_yaml}. Enable pipeline.prepare_dataset or create the dataset first."
        )

    run_name = model_cfg.get("name", "tennis_player_yolo11s")
    weights_name = model_cfg.get("weights") or "yolo11s.pt"
    weights = based_models_dir / weights_name
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing pretrained weights at {weights}. "
            f"Place {weights_name} under {based_models_dir}."
        )

    _configure_mlflow(mlflow_cfg, run_name=run_name)

    logger.info("data_yaml={}", data_yaml)
    logger.info("pretrained_weights={}", weights)
    logger.info("trained_models_dir={}", trained_models_dir)
    logger.info("Starting training name={}", run_name)

    model = YOLO(str(weights))
    results = model.train(
        data=str(data_yaml),
        epochs=int(train_cfg.get("epochs", 100)),
        imgsz=int(train_cfg.get("imgsz", 640)),
        batch=int(train_cfg.get("batch", 16)),
        device=train_cfg.get("device", "0"),
        seed=int(dataset_cfg.get("split_seed", 42)),
        project=str(trained_models_dir),
        name=run_name,
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best weights missing at {best_pt}")

    stable_path = trained_models_dir / f"{run_name}.pt"
    shutil.copy2(best_pt, stable_path)
    logger.info("Saved stable weights to {}", stable_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tennis player detection (YOLO)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "train.yaml"),
        help="Path to the training config YAML",
    )
    args = parser.parse_args()
    run(load_config(args.config))
