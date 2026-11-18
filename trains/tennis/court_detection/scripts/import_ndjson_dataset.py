"""Import an Ultralytics platform NDJSON pose dataset into a local YOLO-pose layout."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from loguru import logger

# Keep in sync with tools/court_schema.FLIP_IDX (14 kpts L/R).
FLIP_IDX = [3, 2, 1, 0, 6, 5, 4, 9, 8, 7, 13, 12, 11, 10]
N_KEYPOINTS = 14
VALID_SPLITS = ("train", "val", "test")


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning("Download failed for {}: {}", dest.name, exc)
        return False


def _format_pose_line(pose: list) -> str:
    """Format [cls, xc, yc, w, h, x,y,v...] with cls as int."""
    if len(pose) < 5 + N_KEYPOINTS * 3:
        raise ValueError(f"Pose annotation too short: len={len(pose)}")
    parts: list[str] = [str(int(pose[0]))]
    for value in pose[1:]:
        parts.append(f"{float(value):.6f}")
    return " ".join(parts)


def import_ndjson_dataset(
    ndjson_path: Path,
    dataset_dir: Path,
    *,
    class_names: list[str] | None = None,
    skip_existing: bool = True,
    include_test: bool = True,
) -> Path:
    """
    Download images and write YOLO-pose labels from an Ultralytics NDJSON export.

    Layout: ``dataset_dir/images/{split}/``, ``dataset_dir/labels/{split}/``, ``data.yaml``.
    """
    ndjson_path = Path(ndjson_path)
    dataset_dir = Path(dataset_dir)
    if not ndjson_path.is_file():
        raise FileNotFoundError(f"NDJSON not found: {ndjson_path}")

    class_names = class_names or ["tennis court"]
    meta_kpt_shape: list[int] | None = None

    n_ok = n_skip = n_fail = n_bad = 0
    splits_count = {s: 0 for s in VALID_SPLITS}

    for split in VALID_SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    with ndjson_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON at line {}: {}", line_no, exc)
                n_bad += 1
                continue

            if record.get("type") == "dataset":
                meta_kpt_shape = record.get("kpt_shape")
                names = record.get("class_names")
                if isinstance(names, dict) and names:
                    class_names = [names[str(i)] for i in range(len(names))]
                continue

            if record.get("type") != "image":
                continue

            split = str(record.get("split", "train"))
            if split not in VALID_SPLITS:
                logger.warning("Unknown split {!r} at line {}, skipping", split, line_no)
                n_bad += 1
                continue
            if split == "test" and not include_test:
                continue

            filename = record.get("file")
            url = record.get("url")
            pose_list = (record.get("annotations") or {}).get("pose") or []
            if not filename or not url or not pose_list:
                logger.warning("Incomplete image record at line {}", line_no)
                n_bad += 1
                continue

            image_path = dataset_dir / "images" / split / str(filename)
            label_path = dataset_dir / "labels" / split / f"{Path(str(filename)).stem}.txt"

            if skip_existing and image_path.exists() and label_path.exists():
                n_skip += 1
                splits_count[split] += 1
                continue

            if not image_path.exists() or not skip_existing:
                if not _download(str(url), image_path):
                    n_fail += 1
                    continue

            try:
                label_line = _format_pose_line(pose_list[0])
            except (ValueError, TypeError, IndexError) as exc:
                logger.warning("Bad pose at line {}: {}", line_no, exc)
                n_bad += 1
                continue

            label_path.write_text(label_line + "\n", encoding="utf-8")
            n_ok += 1
            splits_count[split] += 1

    if meta_kpt_shape and list(meta_kpt_shape) != [N_KEYPOINTS, 3]:
        logger.warning(
            "NDJSON kpt_shape={!r} but importer expects [{}, 3]",
            meta_kpt_shape,
            N_KEYPOINTS,
        )

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [N_KEYPOINTS, 3],
        "flip_idx": FLIP_IDX,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    if include_test and splits_count["test"] > 0:
        data_yaml["test"] = "images/test"

    out_yaml = dataset_dir / "data.yaml"
    with out_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    logger.info(
        "NDJSON import done — written/updated: {}, skipped: {}, failed: {}, bad: {} → {}",
        n_ok,
        n_skip,
        n_fail,
        n_bad,
        out_yaml,
    )
    logger.info("Split counts: {}", splits_count)
    if splits_count["train"] == 0:
        raise RuntimeError("Import produced 0 train images")
    return out_yaml


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import Ultralytics court pose NDJSON to YOLO")
    parser.add_argument("--ndjson", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    import_ndjson_dataset(
        Path(args.ndjson),
        Path(args.dataset_dir),
        skip_existing=not args.no_skip_existing,
    )
