"""Run a trained court pose model on an image and show keypoints + court overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.models.tennis.tennis_court_model import predict_court_keypoints  # noqa: E402
from pipeline.objects.tennis.tennis_court_schema import (  # noqa: E402
    N_POINTS,
    POINT_BY_ID,
    TennisCourtSchema,
    overlay_segments_m,
)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    buf.tofile(str(path))


def extract_keypoints(
    model: YOLO,
    image: np.ndarray,
    *,
    device: str = "cpu",
    imgsz: int = 640,
    stretch: bool = True,
    min_conf: float = 0.2,
) -> dict[int, tuple[float, float]]:
    """Best court instance keypoints in image pixels (same preprocessing as the pipeline)."""
    prediction = predict_court_keypoints(
        model, image, device=device, input_size=(imgsz, imgsz), stretch=stretch
    )
    if prediction is None:
        return {}
    pts, confs = prediction
    out: dict[int, tuple[float, float]] = {}
    for pid in range(min(N_POINTS, len(pts))):
        x, y = float(pts[pid][0]), float(pts[pid][1])
        if float(confs[pid]) < min_conf:
            continue
        if x <= 0 and y <= 0:
            continue
        out[pid] = (x, y)
    return out


def draw_homography_overlay(image: np.ndarray, points: dict[int, tuple[float, float]]) -> np.ndarray:
    overlay = image.copy()
    if len(points) < 4:
        logger.warning("Fewer than 4 keypoints; drawing points only (no homography)")
    else:
        pids = sorted(points)
        image_pts = [points[pid] for pid in pids]
        h_mat = TennisCourtSchema().estimate_court_to_image(image_pts, pids)
        if h_mat is not None:
            logger.info("Homography OK — {} points", len(pids))
            for (x0, y0), (x1, y1) in overlay_segments_m():
                pts = np.array([[[x0, y0], [x1, y1]]], dtype=np.float64)
                proj = cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)
                p0 = (int(round(proj[0, 0])), int(round(proj[0, 1])))
                p1 = (int(round(proj[1, 0])), int(round(proj[1, 1])))
                cv2.line(overlay, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            logger.error("Homography estimation failed")

    for pid, (x, y) in points.items():
        cv2.drawMarker(
            overlay,
            (int(round(x)), int(round(y))),
            (0, 215, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        name = POINT_BY_ID[pid].name if pid in POINT_BY_ID else str(pid)
        cv2.putText(
            overlay,
            f"{pid}:{name}",
            (int(round(x)) + 6, int(round(y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tennis court keypoints + overlay")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output image path (default: <stem>_court_pred.png next to input)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--no-stretch",
        action="store_true",
        help="Letterbox instead of stretching (only for a model trained on letterboxed images)",
    )
    parser.add_argument("--show", action="store_true", help="Open an OpenCV window")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    image = imread_unicode(image_path)
    model = YOLO(str(weights))
    points = extract_keypoints(
        model, image, device=args.device, imgsz=args.imgsz, stretch=not args.no_stretch
    )
    logger.info("Detected {}/{} keypoints", len(points), N_POINTS)

    overlay = draw_homography_overlay(image, points)
    out_path = Path(args.out) if args.out else image_path.with_name(f"{image_path.stem}_court_pred.png")
    imwrite_unicode(out_path, overlay)
    logger.info("Wrote {}", out_path)

    if args.show:
        cv2.imshow("Court pose prediction", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
