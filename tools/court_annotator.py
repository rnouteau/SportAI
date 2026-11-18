"""Annotate 14 tennis court keypoints on a still image and verify via homography overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tkinter import Canvas, Event, Tk

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image, ImageTk

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
for _path in (_REPO_ROOT, _TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from court_yolo import load_pose_label_points, write_pose_label  # noqa: E402
from pipeline.objects.tennis.tennis_court_schema import (  # noqa: E402
    EDGES,
    N_POINTS,
    POINT_BY_ID,
    POINTS,
    TennisCourtSchema,
    model_xy,
    overlay_segments_m,
)

CANVAS_W = 1280
CANVAS_H = 720
HIT_RADIUS_PX = 12
CROSS_HALF = 8


def imread_unicode(path: Path) -> np.ndarray:
    """Load an image with OpenCV even when the path contains non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


class CourtAnnotatorApplication:
    def __init__(self, config: dict, root: Tk) -> None:
        self.root = root
        self.image_path = Path(config["image_path"]).resolve()
        self.bgr = imread_unicode(self.image_path)
        self.img_h, self.img_w = self.bgr.shape[:2]
        self.offset_x = (CANVAS_W - self.img_w) // 2
        self.offset_y = (CANVAS_H - self.img_h) // 2

        # id -> (x, y) in image pixel coordinates
        self.annotations: dict[int, tuple[float, float]] = {}
        self.active_id = 0

        self.root.title("Court Annotator Tool")
        self.root.geometry(f"{CANVAS_W + 2}x{CANVAS_H + 52}")
        self.root.resizable(False, False)

        self.image_panel = Canvas(self.root, width=CANVAS_W, height=CANVAS_H, highlightthickness=0)
        canvas_rgb = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        y0, x0 = max(0, self.offset_y), max(0, self.offset_x)
        y1, x1 = y0 + min(self.img_h, CANVAS_H), x0 + min(self.img_w, CANVAS_W)
        src = rgb[: y1 - y0, : x1 - x0]
        canvas_rgb[y0:y1, x0:x1] = src
        self.base_photo = ImageTk.PhotoImage(Image.fromarray(canvas_rgb))
        self.bg_id = self.image_panel.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.base_photo)
        self.image_panel.place(x=0, y=0)

        self.info_panel = Canvas(self.root, bg="black", width=CANVAS_W, height=50, highlightthickness=0)
        self.info_text_id = self.info_panel.create_text(
            20,
            25,
            anchor="w",
            text="",
            font=("Arial", 12),
            fill="white",
        )
        self.info_panel.place(x=0, y=CANVAS_H)

        self.directory = self.image_path.parent
        self.image_name = self.image_path.stem
        self.txt_path = self.directory / f"{self.image_name}.txt"
        self.csv_path = self.directory / f"{self.image_name}_court.csv"

        self.root.bind("q", self.close_window)
        self.root.bind("d", self.delete_active)
        self.root.bind("s", self.export_annotations)
        self.root.bind("v", self.verify_overlay)
        self.root.bind("<Left>", lambda e: self.shift_active(-1))
        self.root.bind("<Right>", lambda e: self.shift_active(1))
        self.image_panel.bind("<Button-1>", self.on_click)
        self.root.focus_set()

        self.import_annotations()
        self.redraw()

    def canvas_to_image(self, cx: float, cy: float) -> tuple[float, float] | None:
        ix = cx - self.offset_x
        iy = cy - self.offset_y
        if ix < 0 or iy < 0 or ix >= self.img_w or iy >= self.img_h:
            return None
        return (float(ix), float(iy))

    def image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return (ix + self.offset_x, iy + self.offset_y)

    def shift_active(self, delta: int) -> None:
        self.active_id = (self.active_id + delta) % N_POINTS
        self.update_info()
        self.redraw()

    def find_nearest_point(self, ix: float, iy: float) -> int | None:
        best_id: int | None = None
        best_d2 = HIT_RADIUS_PX * HIT_RADIUS_PX
        for pid, (x, y) in self.annotations.items():
            d2 = (x - ix) ** 2 + (y - iy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_id = pid
        return best_id

    def on_click(self, event: Event) -> None:
        coords = self.canvas_to_image(event.x, event.y)
        if coords is None:
            return
        ix, iy = coords

        near = self.find_nearest_point(ix, iy)
        if near is not None and near != self.active_id:
            self.active_id = near
            self.update_info()
            self.redraw()
            return

        self.annotations[self.active_id] = (ix, iy)
        if self.active_id < N_POINTS - 1 and (self.active_id + 1) not in self.annotations:
            self.active_id += 1
        self.redraw()

    def delete_active(self, event: Event | None = None) -> None:
        if self.active_id in self.annotations:
            del self.annotations[self.active_id]
            logger.info(f"Deleted point {self.active_id} ({POINT_BY_ID[self.active_id].name})")
        self.redraw()

    def update_info(self) -> None:
        p = POINT_BY_ID[self.active_id]
        placed = len(self.annotations)
        status = "placed" if self.active_id in self.annotations else "missing"
        self.info_panel.itemconfig(
            self.info_text_id,
            text=(
                f"active {p.id}:{p.name} ({status}) — {p.label} | "
                f"{placed}/{N_POINTS} | ←/→ select  click place  d delete  "
                f"s save (.txt+.csv)  v verify  q quit"
            ),
        )

    def redraw(self) -> None:
        self.image_panel.delete("anno")
        for a, b in EDGES:
            if a in self.annotations and b in self.annotations:
                ax, ay = self.image_to_canvas(*self.annotations[a])
                bx, by = self.image_to_canvas(*self.annotations[b])
                self.image_panel.create_line(
                    ax, ay, bx, by, fill="#33cc66", width=2, tags=("anno",)
                )

        for pid, (ix, iy) in self.annotations.items():
            cx, cy = self.image_to_canvas(ix, iy)
            color = "#ffcc00" if pid == self.active_id else "#3399ff"
            self.image_panel.create_line(
                cx, cy - CROSS_HALF, cx, cy + CROSS_HALF, width=3, fill=color, tags=("anno",)
            )
            self.image_panel.create_line(
                cx - CROSS_HALF, cy, cx + CROSS_HALF, cy, width=3, fill=color, tags=("anno",)
            )
            name = POINT_BY_ID[pid].name
            self.image_panel.create_text(
                cx + 10,
                cy - 10,
                text=f"{pid}:{name}",
                anchor="sw",
                fill=color,
                font=("Arial", 10, "bold"),
                tags=("anno",),
            )

        if self.active_id not in self.annotations:
            # show target hint only in info bar
            pass
        self.update_info()

    def import_annotations(self) -> None:
        if self.txt_path.exists():
            logger.info(f"Import YOLO-pose label from {self.txt_path}")
            try:
                self.annotations = load_pose_label_points(
                    self.txt_path, self.img_w, self.img_h
                )
            except ValueError as exc:
                logger.error(f"Could not parse {self.txt_path}: {exc}")
                self.annotations = {}
        elif self.csv_path.exists():
            logger.info(f"Import legacy CSV from {self.csv_path}")
            data = pd.read_csv(self.csv_path)
            for row in data.itertuples(index=False):
                try:
                    pid = int(getattr(row, "id"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if pid not in POINT_BY_ID:
                    continue
                x, y = getattr(row, "x", None), getattr(row, "y", None)
                if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
                    continue
                self.annotations[pid] = (float(x), float(y))
        else:
            return

        if self.annotations:
            missing = [p.id for p in POINTS if p.id not in self.annotations]
            self.active_id = missing[0] if missing else 0

    def export_annotations(self, event: Event | None = None) -> None:
        if not self.annotations:
            logger.warning("Nothing to save")
            return
        if len(self.annotations) < N_POINTS:
            logger.warning(f"Partial save: {len(self.annotations)}/{N_POINTS} points")

        try:
            write_pose_label(self.txt_path, self.annotations, self.img_w, self.img_h)
        except ValueError as exc:
            logger.error(f"Could not write YOLO label: {exc}")
            return
        logger.info(f"Saved YOLO-pose label to {self.txt_path}")

        rows = []
        for p in POINTS:
            if p.id not in self.annotations:
                continue
            x, y = self.annotations[p.id]
            rows.append({"id": p.id, "name": p.name, "x": x, "y": y})
        df = pd.DataFrame(rows, columns=["id", "name", "x", "y"])
        df.to_csv(self.csv_path, index=False)
        logger.info(f"Saved secondary CSV to {self.csv_path}")

    def verify_overlay(self, event: Event | None = None) -> None:
        if len(self.annotations) < 4:
            logger.error("Need at least 4 annotated points for homography")
            return

        pids = sorted(self.annotations)
        image_pts = [self.annotations[pid] for pid in pids]
        h_mat = TennisCourtSchema().estimate_court_to_image(image_pts, pids)
        if h_mat is None:
            logger.error("Homography estimation failed")
            return

        src_arr = np.asarray([list(model_xy(pid)) for pid in pids], dtype=np.float64)
        dst_arr = np.asarray(image_pts, dtype=np.float64)
        projected = cv2.perspectiveTransform(src_arr.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
        mean_err = float(np.mean(np.linalg.norm(projected - dst_arr, axis=1)))
        logger.info(
            f"Homography OK — {len(pids)} points, mean reprojection error {mean_err:.2f}px"
        )

        overlay = self.bgr.copy()
        for (x0, y0), (x1, y1) in overlay_segments_m():
            pts = np.array([[[x0, y0], [x1, y1]]], dtype=np.float64)
            proj = cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)
            p0 = (int(round(proj[0, 0])), int(round(proj[0, 1])))
            p1 = (int(round(proj[1, 0])), int(round(proj[1, 1])))
            cv2.line(overlay, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)

        for pid, (ix, iy) in self.annotations.items():
            color = (0, 215, 255) if pid == self.active_id else (255, 128, 0)
            cv2.drawMarker(
                overlay,
                (int(round(ix)), int(round(iy))),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )

        win = "Court overlay (press any key)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, overlay)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    def close_window(self, event: Event | None = None) -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Court Annotator Tool (14 ITF keypoints)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    root = Tk()
    CourtAnnotatorApplication({"image_path": args.image_path}, root)
    root.mainloop()
