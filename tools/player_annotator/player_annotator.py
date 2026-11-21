"""Player annotator with track ids (ground truth for detection *and* tracking).

    python -m tools.player_annotator --video_path path/to/clip.mp4

Without an existing ``<clip>_players.csv`` the clip is pre-annotated with the player
model + ByteTrack (ids 1 / 2, 1 = player lowest in the image on the first frame).
Pre-annotated boxes are dashed; boxes you drew or validated (keyframes) are solid.

Keys
    Right / Left          next / previous frame (Right marks the current frame reviewed)
    Shift+Right / Left    jump 10 frames (does not mark anything)
    1..9                  mouse over a box: give it that id (swaps with the box that had it)
                          otherwise: current id for new boxes
    drag (left button)    draw a box with the current id (replaces that id on this frame)
    d                     delete the box under the mouse
    w / W                 swap ids 1 <-> 2 on this frame / from this frame to the end
    c                     copy the boxes of the previous annotated frame
    k                     validate every box of this frame as a keyframe
    i                     interpolate each id from its previous keyframe to this frame
    v                     toggle "reviewed" (use it for frames with no player: close-ups...)
    s                     save CSV        e   save CSV + export MOT (gt/gt.txt)
    h                     show / hide help                      q   save and quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tkinter import Canvas, Event, Tk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from player_annotations import UNKNOWN_ID, PlayerAnnotations, PlayerBox, TrackIdMapper

CANVAS_W, CANVAS_H = 1280, 720
INFO_H = 70
JUMP = 10
MIN_BOX_PX = 5  # drags smaller than this (canvas pixels) are ignored
ID_COLORS = {1: "#ff3b30", 2: "#0a84ff", 3: "#30d158", 4: "#ff9f0a", 5: "#bf5af2"}
UNKNOWN_COLOR = "#ffd60a"
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "trains" / "tennis" / "player_detection" / "models" / "based_models" / "yolo11s.pt"  #/ "models" / "TENNIS_PLAYER_MODEL_17_11_2026.pt"
)
HELP_SHORT = (
    "→/← frame (→ = vérifiée)  Maj+→/← ±10  1-9 ID / réassigner  glisser = boîte  d suppr  "
    "w/W échanger  c copier  k valider  i interpoler  v vérifiée  s sauver  e MOT  h aide  q quitter"
)


def color_for(track_id: int) -> str:
    return ID_COLORS.get(track_id, UNKNOWN_COLOR)


def _to_numpy(values: Any) -> np.ndarray:
    """Ultralytics returns torch tensors (or numpy arrays): always give numpy."""
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    return np.asarray(values)


def pre_annotate(
    video_path: str,
    weights: str | Path,
    *,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.1,
    device: str | None = None,
    n_players: int = 2,
) -> PlayerAnnotations:
    """Detect + track people on the whole clip and keep ``n_players`` ids (not reviewed)."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing player weights at {weights} (use --weights).")
    model = YOLO(str(weights))
    mapper = TrackIdMapper(n_players=n_players)
    ann = PlayerAnnotations()
    kwargs: dict[str, Any] = {
        "classes": [0],
        "stream": True,
        "verbose": False,
        "persist": True,
        "tracker": tracker,
        "conf": conf,
    }
    if device:
        kwargs["device"] = device
    for frame_idx, result in enumerate(model.track(video_path, **kwargs)):
        boxes = result.boxes
        if frame_idx % 100 == 0:
            print(f"pre-annotation: frame {frame_idx}")
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        ids = [int(v) for v in _to_numpy(boxes.id)] if boxes.id is not None else [None] * len(xyxy)
        detections = [(ids[i], float(confs[i]), xyxy[i].tolist()) for i in range(len(xyxy))]
        for box in mapper.update(detections):
            ann.set_box(frame_idx, box, review=False)
    return ann


class PlayerAnnotatorApp:
    def __init__(
        self,
        root: Tk,
        video_path: str,
        *,
        weights: str | Path = DEFAULT_WEIGHTS,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.1,
        device: str | None = None,
        reannotate: bool = False,
    ) -> None:
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = min(CANVAS_W / self.video_w, CANVAS_H / self.video_h)

        path = Path(video_path)
        self.directory = path.parent.absolute()
        self.video_name = path.name.removesuffix("".join(path.suffixes))
        self.csv_path = self.directory / f"{self.video_name}_players.csv"
        self.mot_dir = self.directory / f"{self.video_name}_mot"

        if self.csv_path.exists() and not reannotate:
            print(f"import annotations from {self.csv_path}")
            self.ann = PlayerAnnotations.load_csv(self.csv_path)
        else:
            print(f"pre-annotate {video_path}")
            self.ann = pre_annotate(video_path, weights, tracker=tracker, conf=conf, device=device)

        self.current_frame = 0
        self.current_id = 1
        self.mouse = (0.0, 0.0)
        self.drag_start: tuple[float, float] | None = None
        self.status = ""
        self.show_help = False
        self.photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.show_current_frame()

    # ------------------------------------------------------------------ UI setup
    def _build_ui(self) -> None:
        self.root.title(f"Player Annotator — {self.video_name}")
        self.root.geometry(f"{CANVAS_W}x{CANVAS_H + INFO_H}")
        self.root.resizable(False, False)
        self.canvas = Canvas(
            self.root, width=CANVAS_W, height=CANVAS_H, bg="black", highlightthickness=0
        )
        self.canvas.pack()
        self.info = Canvas(
            self.root, width=CANVAS_W, height=INFO_H, bg="black", highlightthickness=0
        )
        self.info.pack()
        self.bg_id = self.canvas.create_image(0, 0, anchor="nw")
        self.info_line = self.info.create_text(
            10, 16, anchor="w", fill="white", font=("Arial", 12, "bold")
        )
        self.status_line = self.info.create_text(
            10, 38, anchor="w", fill="#ffd60a", font=("Arial", 11)
        )
        self.help_line = self.info.create_text(
            10, 58, anchor="w", fill="#aaaaaa", font=("Arial", 9), text=HELP_SHORT
        )

        self.root.bind("<Right>", lambda e: self.step(1, mark=True))
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Shift-Right>", lambda e: self.step(JUMP))
        self.root.bind("<Shift-Left>", lambda e: self.step(-JUMP))
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ------------------------------------------------------------------ coords
    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        x = min(max(cx / self.scale, 0.0), float(self.video_w))
        y = min(max(cy / self.scale, 0.0), float(self.video_h))
        return x, y

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    # ------------------------------------------------------------------ rendering
    def read_frame(self, index: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        return frame if ok else None

    def show_current_frame(self) -> None:
        frame = self.read_frame(self.current_frame)
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.scale != 1.0:
                size = (round(self.video_w * self.scale), round(self.video_h * self.scale))
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.canvas.itemconfig(self.bg_id, image=self.photo)
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("ann")
        for box in self.ann.frame_boxes(self.current_frame):
            x0, y0 = self.to_canvas(box.x0, box.y0)
            x1, y1 = self.to_canvas(box.x1, box.y1)
            color = color_for(box.track_id)
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=color,
                width=3 if box.keyframe else 2,
                dash=() if box.keyframe else (6, 4),
                tags=("ann",),
            )
            label = "?" if box.track_id == UNKNOWN_ID else f"P{box.track_id}"
            self.canvas.create_text(
                x0 + 2,
                y0 - 2,
                anchor="sw",
                text=label,
                fill=color,
                font=("Arial", 12, "bold"),
                tags=("ann",),
            )
        if self.show_help:
            self.canvas.create_rectangle(
                20, 20, 760, 420, fill="black", stipple="gray50", outline="white", tags=("ann",)
            )
            self.canvas.create_text(
                36,
                32,
                anchor="nw",
                fill="white",
                font=("Consolas", 11),
                text=(__doc__ or "").split("Keys", 1)[-1],
                tags=("ann",),
            )
        self.update_info()

    def update_info(self) -> None:
        reviewed = self.ann.is_reviewed(self.current_frame)
        text = (
            f"frame {self.current_frame}/{self.frame_count - 1}    "
            f"ID courant : P{self.current_id}    "
            f"{'✔ vérifiée' if reviewed else '✗ non vérifiée'}    "
            f"vérifiées : {len(self.ann.reviewed)}/{self.frame_count}"
        )
        self.info.itemconfig(self.info_line, text=text, fill="#30d158" if reviewed else "white")
        self.info.itemconfig(self.status_line, text=self.status)

    def set_status(self, message: str) -> None:
        self.status = message
        self.update_info()

    # ------------------------------------------------------------------ navigation
    def step(self, delta: int, *, mark: bool = False) -> None:
        if mark:
            self.ann.mark_reviewed(self.current_frame)
        target = min(max(self.current_frame + delta, 0), self.frame_count - 1)
        self.status = ""
        if target != self.current_frame:
            self.current_frame = target
            self.show_current_frame()
        else:
            self.redraw()

    # ------------------------------------------------------------------ mouse
    def on_motion(self, event: Event) -> None:
        self.mouse = self.to_image(event.x, event.y)

    def on_press(self, event: Event) -> None:
        self.drag_start = (float(event.x), float(event.y))

    def on_drag(self, event: Event) -> None:
        self.mouse = self.to_image(event.x, event.y)
        if self.drag_start is None:
            return
        self.canvas.delete("drag")
        self.canvas.create_rectangle(
            *self.drag_start,
            event.x,
            event.y,
            outline=color_for(self.current_id),
            width=2,
            tags=("drag",),
        )

    def on_release(self, event: Event) -> None:
        self.canvas.delete("drag")
        if self.drag_start is None:
            return
        sx, sy = self.drag_start
        self.drag_start = None
        if abs(event.x - sx) < MIN_BOX_PX or abs(event.y - sy) < MIN_BOX_PX:
            return
        xa, ya = self.to_image(sx, sy)
        xb, yb = self.to_image(event.x, event.y)
        box = PlayerBox.from_corners(self.current_id, xa, ya, xb, yb, keyframe=True)
        self.ann.set_box(self.current_frame, box)
        self.status = f"boîte P{self.current_id} ajoutée"
        self.redraw()

    # ------------------------------------------------------------------ keys
    def on_key(self, event: Event) -> None:
        char = event.char or ""
        frame = self.current_frame
        x, y = self.mouse
        if char.isdigit() and char != "0":
            new_id = int(char)
            if self.ann.reassign_at(frame, x, y, new_id):
                self.status = f"boîte réassignée à P{new_id}"
            else:
                self.current_id = new_id
                self.status = f"ID courant : P{new_id}"
        elif char == "d":
            self.status = "boîte supprimée" if self.ann.delete_at(frame, x, y) else ""
        elif char == "w":
            n = self.ann.swap_ids(frame, 1, 2)
            self.status = "P1 ↔ P2 sur cette frame" if n else "rien à échanger"
        elif char == "W":
            n = self.ann.swap_ids(frame, 1, 2, to_end=True)
            self.status = f"P1 ↔ P2 sur {n} frames (jusqu'à la fin)"
        elif char == "c":
            ok = self.ann.copy_previous(frame)
            self.status = "boîtes copiées depuis la frame précédente" if ok else "rien à copier"
        elif char == "k":
            n = self.ann.mark_keyframes(frame)
            self.status = f"{n} boîte(s) validée(s) comme keyframe"
        elif char == "i":
            n = self.ann.interpolate_to(frame)
            self.status = f"{n} boîte(s) interpolée(s)" if n else "pas de keyframe précédente"
        elif char == "v":
            state = self.ann.toggle_reviewed(frame)
            self.status = "frame vérifiée" if state else "frame non vérifiée"
        elif char == "s":
            self.save()
        elif char == "e":
            self.export_mot()
        elif char == "h":
            self.show_help = not self.show_help
        elif char == "q":
            self.quit()
            return
        else:
            return
        self.redraw()

    # ------------------------------------------------------------------ files
    def save(self) -> None:
        path = self.ann.save_csv(self.csv_path)
        print(f"save annotations to {path}")
        self.status = f"sauvegardé : {path.name}"

    def export_mot(self) -> None:
        self.save()
        gt_path, n_boxes, skipped = self.ann.export_mot(self.mot_dir)
        missing = len(self.ann.unreviewed_frames(self.frame_count))
        print(f"MOT export: {gt_path} ({n_boxes} boxes, {skipped} unknown-id skipped)")
        self.status = (
            f"MOT : {n_boxes} boîtes -> {gt_path.parent.parent.name}/gt/gt.txt"
            + (f"  |  {skipped} boîte(s) sans ID ignorée(s)" if skipped else "")
            + (f"  |  {missing} frame(s) non vérifiée(s)" if missing else "")
        )

    def quit(self) -> None:
        self.save()
        self.cap.release()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Player annotator (boxes + track ids)")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS), help="Player model for pre-annotation"
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker yaml")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection threshold for tracking")
    parser.add_argument("--device", default=None, help="cpu, cuda, 0...")
    parser.add_argument(
        "--reannotate",
        action="store_true",
        help="Ignore the existing CSV and pre-annotate again (overwritten on save)",
    )
    args = parser.parse_args()

    root = Tk()
    PlayerAnnotatorApp(
        root,
        args.video_path,
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        device=args.device,
        reannotate=args.reannotate,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
