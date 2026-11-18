"""Build an Ultralytics YOLO-pose dataset from court annotator images + CSV."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Must match tools/court_schema.py ids 0..13 (left-right swap for hflip aug).
N_KEYPOINTS = 14
FLIP_IDX = [3, 2, 1, 0, 6, 5, 4, 9, 8, 7, 13, 12, 11, 10]

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


def _load_keypoints(csv_path: Path) -> dict[int, tuple[float, float]]:
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


def _bbox_from_keypoints(
    points: dict[int, tuple[float, float]],
    width: int,
    height: int,
    padding: float,
) -> tuple[float, float, float, float] | None:
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = padding * width
    pad_y = padding * height
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _xyxy_to_yolo(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xc = ((x0 + x1) / 2.0) / width
    yc = ((y0 + y1) / 2.0) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return (
        float(min(max(xc, 0.0), 1.0)),
        float(min(max(yc, 0.0), 1.0)),
        float(min(max(bw, 0.0), 1.0)),
        float(min(max(bh, 0.0), 1.0)),
    )


def _pose_label_line(
    points: dict[int, tuple[float, float]],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    xc, yc, bw, bh = _xyxy_to_yolo(*bbox, width, height)
    parts = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]
    for pid in range(N_KEYPOINTS):
        if pid in points:
            x, y = points[pid]
            parts.append(f"{x / width:.6f} {y / height:.6f} 2")
        else:
            parts.append("0.000000 0.000000 0")
    return " ".join(parts)


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
    bbox_padding: float = 0.02,
) -> Path:
    """
    Convert still images + ``*_court.csv`` into a YOLO-pose train/val dataset.

    Split is by image file (seeded). Requires at least ``min_keypoints`` visible points.
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
        csv_path = image_path.with_name(f"{image_path.stem}{annotations_suffix}")
        if not csv_path.exists():
            logger.warning("Skipping {}: missing {}", image_path.name, csv_path.name)
            continue
        points = _load_keypoints(csv_path)
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
            f"(need '*{annotations_suffix}' with >= {min_keypoints} points)"
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
        bbox = _bbox_from_keypoints(points, width, height, bbox_padding)
        if bbox is None:
            logger.warning("Skipping {}: invalid bbox", image_path.name)
            continue

        out_ext = image_path.suffix.lower()
        stem = image_path.stem
        out_image = dataset_dir / "images" / split / f"{stem}{out_ext}"
        out_label = dataset_dir / "labels" / split / f"{stem}.txt"
        if not cv2.imwrite(str(out_image), frame):
            # Fallback for odd paths: encode + tofile
            ok, buf = cv2.imencode(out_ext, frame)
            if not ok:
                raise RuntimeError(f"Failed to write image {out_image}")
            buf.tofile(str(out_image))

        out_label.write_text(_pose_label_line(points, bbox, width, height) + "\n", encoding="utf-8")
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
