"""Build an Ultralytics YOLO-pose dataset from court images + YOLO .txt and/or CSV."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.objects.tennis.tennis_court_schema import FLIP_IDX, N_POINTS  # noqa: E402
from tools.court_annotator.court_yolo import (  # noqa: E402
    DEFAULT_BBOX_PADDING,
    bbox_from_keypoints,
    encode_pose_label_line,
    load_pose_label_points,
)

N_KEYPOINTS = N_POINTS
IMAGE_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp")


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _split_units(units: list, val_ratio: float, seed: int) -> tuple[set, set]:
    if not units:
        return set(), set()
    rng = random.Random(seed)
    ordered = list(units)
    rng.shuffle(ordered)
    if len(ordered) == 1:
        logger.warning("Only one image available; assigning it to train")
        return {ordered[0]}, set()
    n_val = max(1, int(round(len(ordered) * val_ratio)))
    n_val = min(n_val, len(ordered) - 1)
    return set(ordered[n_val:]), set(ordered[:n_val])


def _reset_dataset_dirs(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _load_keypoints_csv(csv_path: Path) -> dict[int, tuple[float, float]]:
    data = pd.read_csv(csv_path)
    points: dict[int, tuple[float, float]] = {}
    if "id" not in data.columns or "x" not in data.columns or "y" not in data.columns:
        logger.warning("Skipping {}: expected columns id,x,y", csv_path.name)
        return points
    for row in data.itertuples(index=False):
        try:
            pid = int(row.id)
            x, y = float(row.x), float(row.y)
        except (TypeError, ValueError):
            continue
        if pid < 0 or pid >= N_KEYPOINTS:
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        points[pid] = (x, y)
    return points


def _resolve_annotation_points(
    image_path: Path,
    width: int,
    height: int,
    annotations_suffix: str,
) -> dict[int, tuple[float, float]] | None:
    """Prefer sibling YOLO ``.txt``, else legacy ``*_court.csv``."""
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            return load_pose_label_points(txt_path, width, height)
        except ValueError as exc:
            logger.warning("Skipping {}: invalid YOLO label ({})", image_path.name, exc)
            return None

    csv_path = image_path.with_name(f"{image_path.stem}{annotations_suffix}")
    if csv_path.exists():
        return _load_keypoints_csv(csv_path)

    logger.warning(
        "Skipping {}: missing {}.txt or *{}",
        image_path.name,
        image_path.stem,
        annotations_suffix,
    )
    return None


def _write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [N_KEYPOINTS, 3],
        "flip_idx": FLIP_IDX,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    out_path = dataset_dir / "data.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)
    return out_path


def prepare_dataset(
    images_dir: Path,
    dataset_dir: Path,
    *,
    class_names: list[str] | None = None,
    annotations_suffix: str = "_court.csv",
    image_exts: list[str] | tuple[str, ...] | None = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    min_keypoints: int = 4,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
) -> Path:
    """
    Convert still images + YOLO-pose ``.txt`` (preferred) or ``*_court.csv`` into train/val.
    """
    images_dir = Path(images_dir)
    dataset_dir = Path(dataset_dir)
    class_names = class_names or ["tennis court"]
    exts = tuple(e if e.startswith(".") else f".{e}" for e in (image_exts or IMAGE_EXTS_DEFAULT))

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    candidates: list[Path] = []
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No images with extensions {exts} under {images_dir}")

    samples: list[tuple[Path, dict[int, tuple[float, float]]]] = []
    for image_path in candidates:
        frame = imread_unicode(image_path)
        if frame is None:
            logger.warning("Could not read {}", image_path)
            continue
        height, width = frame.shape[:2]
        points = _resolve_annotation_points(image_path, width, height, annotations_suffix)
        if points is None:
            continue
        if len(points) < min_keypoints:
            logger.warning(
                "Skipping {}: only {}/{} keypoints (min={})",
                image_path.name,
                len(points),
                N_KEYPOINTS,
                min_keypoints,
            )
            continue
        samples.append((image_path, points))

    if not samples:
        raise FileNotFoundError(
            f"No usable annotated images under {images_dir} "
            f"(need sibling '.txt' or '*{annotations_suffix}' with >= {min_keypoints} points)"
        )

    logger.info("Preparing YOLO-pose dataset from {} image(s)", len(samples))
    _reset_dataset_dirs(dataset_dir)

    train_paths, val_paths = _split_units(
        [p for p, _ in samples],
        val_ratio=val_ratio,
        seed=split_seed,
    )
    logger.info("Image split — train: {}, val: {}", len(train_paths), len(val_paths))

    n_train = n_val = 0
    for image_path, points in samples:
        split = "val" if image_path in val_paths else "train"
        frame = imread_unicode(image_path)
        if frame is None:
            logger.warning("Could not read {}", image_path)
            continue
        height, width = frame.shape[:2]
        bbox = bbox_from_keypoints(points, width, height, bbox_padding)
        if bbox is None:
            logger.warning("Skipping {}: invalid bbox", image_path.name)
            continue

        out_ext = image_path.suffix.lower()
        stem = image_path.stem
        out_image = dataset_dir / "images" / split / f"{stem}{out_ext}"
        out_label = dataset_dir / "labels" / split / f"{stem}.txt"
        if not cv2.imwrite(str(out_image), frame):
            ok, buf = cv2.imencode(out_ext, frame)
            if not ok:
                raise RuntimeError(f"Failed to write image {out_image}")
            buf.tofile(str(out_image))

        out_label.write_text(
            encode_pose_label_line(points, width, height, bbox=bbox, padding=bbox_padding)
            + "\n",
            encoding="utf-8",
        )
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    if n_train == 0:
        raise RuntimeError("Dataset preparation produced 0 train images")
    if n_val == 0:
        logger.warning("Dataset preparation produced 0 val images")

    data_yaml = _write_data_yaml(dataset_dir, class_names)
    logger.info(
        "Dataset ready — train: {} images, val: {} images → {}",
        n_train,
        n_val,
        data_yaml,
    )
    return data_yaml
