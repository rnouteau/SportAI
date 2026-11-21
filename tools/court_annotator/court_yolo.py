"""Shared YOLO-pose encode/decode for tennis court keypoints (14 ITF points)."""

from __future__ import annotations

from pathlib import Path

from pipeline.objects.tennis.tennis_court_schema import N_POINTS

DEFAULT_BBOX_PADDING = 0.02


def bbox_from_keypoints(
    points: dict[int, tuple[float, float]],
    width: int,
    height: int,
    padding: float = DEFAULT_BBOX_PADDING,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bbox (xyxy pixels) around keypoints, with relative padding."""
    if not points:
        return None
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


def xyxy_to_yolo(
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


def encode_pose_label_line(
    points: dict[int, tuple[float, float]],
    width: int,
    height: int,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    padding: float = DEFAULT_BBOX_PADDING,
    class_id: int = 0,
    n_keypoints: int = N_POINTS,
) -> str:
    """One YOLO-pose line: ``cls xc yc w h x y v ...`` (normalized coords)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    box = bbox or bbox_from_keypoints(points, width, height, padding)
    if box is None:
        raise ValueError("Cannot encode pose label without keypoints / bbox")
    xc, yc, bw, bh = xyxy_to_yolo(*box, width, height)
    parts = [f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]
    for pid in range(n_keypoints):
        if pid in points:
            x, y = points[pid]
            parts.append(f"{x / width:.6f} {y / height:.6f} 2")
        else:
            parts.append("0.000000 0.000000 0")
    return " ".join(parts)


def decode_pose_label_line(
    line: str,
    width: int,
    height: int,
    *,
    n_keypoints: int = N_POINTS,
) -> dict[int, tuple[float, float]]:
    """Parse visible keypoints (v > 0) from one YOLO-pose line into pixel coords."""
    parts = line.strip().split()
    expected = 5 + n_keypoints * 3
    if len(parts) < expected:
        raise ValueError(f"Label too short: {len(parts)} < {expected}")
    values = [float(x) for x in parts[:expected]]
    points: dict[int, tuple[float, float]] = {}
    cursor = 5
    for pid in range(n_keypoints):
        x_n, y_n, v = values[cursor], values[cursor + 1], values[cursor + 2]
        cursor += 3
        if int(v) > 0:
            points[pid] = (x_n * width, y_n * height)
    return points


def load_pose_label_points(
    label_path: Path,
    width: int,
    height: int,
    *,
    n_keypoints: int = N_POINTS,
) -> dict[int, tuple[float, float]]:
    text = Path(label_path).read_text(encoding="utf-8").strip()
    if not text:
        return {}
    # First non-empty line
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return {}
    return decode_pose_label_line(line, width, height, n_keypoints=n_keypoints)


def write_pose_label(
    label_path: Path,
    points: dict[int, tuple[float, float]],
    width: int,
    height: int,
    *,
    padding: float = DEFAULT_BBOX_PADDING,
) -> None:
    line = encode_pose_label_line(points, width, height, padding=padding)
    Path(label_path).write_text(line + "\n", encoding="utf-8")
