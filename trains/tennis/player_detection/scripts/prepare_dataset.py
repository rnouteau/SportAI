"""Build an Ultralytics YOLO dataset from annotated tennis videos."""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml
from loguru import logger


def xyxy_to_yolo(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert absolute xyxy box to YOLO normalized xc yc w h."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    bw = max(xb - xa, 0.0) / image_width
    bh = max(yb - ya, 0.0) / image_height
    xc = ((xa + xb) / 2.0) / image_width
    yc = ((ya + yb) / 2.0) / image_height
    return (
        float(min(max(xc, 0.0), 1.0)),
        float(min(max(yc, 0.0), 1.0)),
        float(min(max(bw, 0.0), 1.0)),
        float(min(max(bh, 0.0), 1.0)),
    )


def _split_units(
    units: list,
    val_ratio: float,
    seed: int,
) -> tuple[set, set]:
    """Shuffle units with a fixed seed and carve out a validation subset."""
    if not units:
        return set(), set()

    rng = random.Random(seed)
    ordered = list(units)
    rng.shuffle(ordered)

    if len(ordered) == 1:
        logger.warning("Only one split unit available; assigning it to train")
        return {ordered[0]}, set()

    n_val = max(1, int(round(len(ordered) * val_ratio)))
    n_val = min(n_val, len(ordered) - 1)
    val_units = set(ordered[:n_val])
    train_units = set(ordered[n_val:])
    return train_units, val_units


def _chunk_frames(frames: list[int], chunk_size: int) -> list[tuple[int, ...]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [tuple(frames[i : i + chunk_size]) for i in range(0, len(frames), chunk_size)]


def _load_video_annotations(
    video_path: Path,
    annotations_suffix: str,
) -> dict[int, list[tuple[int, float, float, float, float]]] | None:
    csv_path = video_path.with_name(f"{video_path.stem}{annotations_suffix}")
    if not csv_path.exists():
        logger.warning("Skipping {}: missing {}", video_path.name, csv_path.name)
        return None

    data = pd.read_csv(csv_path).dropna(subset=["class_id", "x0", "y0", "x1", "y1"])
    if data.empty:
        logger.warning("Skipping {}: empty annotations in {}", video_path.name, csv_path.name)
        return None

    annotations: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for row in data.itertuples(index=False):
        annotations[int(row.frame)].append(
            (int(row.class_id), float(row.x0), float(row.y0), float(row.x1), float(row.y1))
        )
    return dict(annotations)


def _reset_dataset_dirs(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_sample(
    *,
    frame_bgr,
    boxes: list[tuple[int, float, float, float, float]],
    dataset_dir: Path,
    split: str,
    stem: str,
    image_ext: str,
) -> None:
    height, width = frame_bgr.shape[:2]
    image_path = dataset_dir / "images" / split / f"{stem}{image_ext}"
    label_path = dataset_dir / "labels" / split / f"{stem}.txt"

    if not cv2.imwrite(str(image_path), frame_bgr):
        raise RuntimeError(f"Failed to write image {image_path}")

    lines: list[str] = []
    for class_id, x0, y0, x1, y1 in boxes:
        xc, yc, bw, bh = xyxy_to_yolo(x0, y0, x1, y1, width, height)
        if bw <= 0.0 or bh <= 0.0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _export_video(
    *,
    video_path: Path,
    annotations: dict[int, list[tuple[int, float, float, float, float]]],
    frame_to_split: dict[int, str],
    dataset_dir: Path,
    image_ext: str,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    n_train = n_val = 0
    try:
        for frame_idx, split in sorted(frame_to_split.items()):
            boxes = annotations.get(frame_idx)
            if not boxes:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                logger.warning("Could not read frame {} from {}", frame_idx, video_path.name)
                continue
            stem = f"{video_path.stem}_{frame_idx:06d}"
            _write_sample(
                frame_bgr=frame_bgr,
                boxes=boxes,
                dataset_dir=dataset_dir,
                split=split,
                stem=stem,
                image_ext=image_ext,
            )
            if split == "train":
                n_train += 1
            else:
                n_val += 1
    finally:
        cap.release()
    return n_train, n_val


def _write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    out_path = dataset_dir / "data.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)
    return out_path


def prepare_dataset(
    videos_dir: Path,
    dataset_dir: Path,
    *,
    class_names: list[str] | None = None,
    annotations_suffix: str = "_players.csv",
    image_ext: str = ".jpg",
    val_ratio: float = 0.2,
    split_seed: int = 42,
    chunk_size: int = 32,
) -> Path:
    """
    Convert annotated videos (+ CSV) into a YOLO train/val dataset.

    Split strategy:
    - >= 2 videos: whole videos are assigned to train or val (seeded shuffle).
    - 1 video: consecutive annotated frames are grouped into chunks, then split.
    """
    videos_dir = Path(videos_dir)
    dataset_dir = Path(dataset_dir)
    class_names = class_names or ["player"]
    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"

    video_paths = sorted(videos_dir.rglob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found under {videos_dir}")

    annotated: list[tuple[Path, dict[int, list[tuple[int, float, float, float, float]]]]] = []
    for video_path in video_paths:
        annotations = _load_video_annotations(video_path, annotations_suffix)
        if annotations:
            annotated.append((video_path, annotations))

    if not annotated:
        raise FileNotFoundError(
            f"No annotated videos found under {videos_dir} "
            f"(expected '*{annotations_suffix}' next to each .mp4)"
        )

    logger.info("Preparing YOLO dataset from {} annotated video(s)", len(annotated))
    _reset_dataset_dirs(dataset_dir)

    total_train = total_val = 0

    if len(annotated) >= 2:
        train_videos, val_videos = _split_units(
            [video for video, _ in annotated],
            val_ratio=val_ratio,
            seed=split_seed,
        )
        logger.info(
            "Video split — train: {}, val: {}",
            len(train_videos),
            len(val_videos),
        )
        for video_path, annotations in annotated:
            split = "val" if video_path in val_videos else "train"
            frame_to_split = {frame_idx: split for frame_idx in annotations}
            n_train, n_val = _export_video(
                video_path=video_path,
                annotations=annotations,
                frame_to_split=frame_to_split,
                dataset_dir=dataset_dir,
                image_ext=image_ext,
            )
            total_train += n_train
            total_val += n_val
    else:
        video_path, annotations = annotated[0]
        frames = sorted(annotations)
        chunks = _chunk_frames(frames, chunk_size)
        train_chunks, val_chunks = _split_units(chunks, val_ratio=val_ratio, seed=split_seed)
        logger.info(
            "Single-video chunk split — train chunks: {}, val chunks: {} (chunk_size={})",
            len(train_chunks),
            len(val_chunks),
            chunk_size,
        )
        frame_to_split: dict[int, str] = {}
        for chunk in train_chunks:
            for frame_idx in chunk:
                frame_to_split[frame_idx] = "train"
        for chunk in val_chunks:
            for frame_idx in chunk:
                frame_to_split[frame_idx] = "val"

        n_train, n_val = _export_video(
            video_path=video_path,
            annotations=annotations,
            frame_to_split=frame_to_split,
            dataset_dir=dataset_dir,
            image_ext=image_ext,
        )
        total_train, total_val = n_train, n_val

    if total_train == 0:
        raise RuntimeError("Dataset preparation produced 0 train images")
    if total_val == 0:
        logger.warning("Dataset preparation produced 0 val images")

    data_yaml = _write_data_yaml(dataset_dir, class_names)
    logger.info(
        "Dataset ready — train: {} images, val: {} images → {}",
        total_train,
        total_val,
        data_yaml,
    )
    return data_yaml
