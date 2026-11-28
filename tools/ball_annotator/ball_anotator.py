"""Ball annotator: ball position and events (serve / hit / bounce) frame by frame.

    python tools/ball_annotator/ball_anotator.py --video_path path/to/clip.mp4

Annotations are saved next to the video in ``<clip>.csv`` (format in ball_annotations.py)
and reloaded when the tool is opened again.

Keys
    left click            ball position on the current frame
    d                     delete the ball of the current frame
    Right / Left          next / previous frame
    Shift+Right / Left    jump 10 frames
    1 / 2 / 3             event serve / hit / bounce on the current frame (again: remove it)
    0                     remove the event of the current frame
    n / p                 go to the next / previous event
    s                     save CSV                q   quit (asks to save if needed)

The frame is scaled to fit the 1280x720 view; positions are stored in video pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tkinter import Canvas, Event, Tk, messagebox

import cv2
from PIL import Image, ImageTk

from ball_annotations import EVENTS, BallAnnotations

CANVAS_W, CANVAS_H = 1280, 720
INFO_H = 60
JUMP = 10
EVENT_KEYS = {"1": "serve", "2": "hit", "3": "bounce"}
EVENT_COLORS = {"serve": "#22c55e", "hit": "#f97316", "bounce": "#e11d48"}
BALL_COLOR = "#3b82f6"


class BallAnnotatorApplication:
    def __init__(self, config: dict, root: Tk) -> None:
        self.root = root
        self.video_path = Path(config["video_path"])
        self.csv_path = self.video_path.with_suffix(".csv")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Video pixels <-> canvas pixels (identity for a 1280x720 video)
        self.scale = min(CANVAS_W / width, CANVAS_H / height)
        self.display_size = (round(width * self.scale), round(height * self.scale))
        self.offset = (
            (CANVAS_W - self.display_size[0]) / 2,
            (CANVAS_H - self.display_size[1]) / 2,
        )

        self.ann = BallAnnotations()
        if self.csv_path.exists():
            self.ann = BallAnnotations.load_csv(self.csv_path)
            print(f"import annotations from {self.csv_path}")
        self.dirty = False
        self.status = ""
        self.current_frame = 0
        self.last_read = -1  # frame index of the last cap.read(), to avoid seeking

        self.root.title(f"Ball Annotator - {self.video_path.name}")
        self.root.geometry(f"{CANVAS_W}x{CANVAS_H + INFO_H}")
        self.root.resizable(False, False)
        self.canvas = Canvas(
            self.root, width=CANVAS_W, height=CANVAS_H, bg="black", highlightthickness=0
        )
        self.canvas.place(x=0, y=0)
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (CANVAS_W, CANVAS_H)))
        self.image_id = self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.photo)
        self.info = Canvas(
            self.root, width=CANVAS_W, height=INFO_H, bg="black", highlightthickness=0
        )
        self.info.place(x=0, y=CANVAS_H)

        self.root.bind("<Right>", lambda e: self.go_to(self.current_frame + 1))
        self.root.bind("<Left>", lambda e: self.go_to(self.current_frame - 1))
        self.root.bind("<Shift-Right>", lambda e: self.go_to(self.current_frame + JUMP))
        self.root.bind("<Shift-Left>", lambda e: self.go_to(self.current_frame - JUMP))
        self.root.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.go_to(0)

    # ---- coordinates -------------------------------------------------------------------

    def to_video(self, cx: float, cy: float) -> tuple[float, float] | None:
        """Canvas click -> video pixels, ``None`` outside the image."""
        x = (cx - self.offset[0]) / self.scale
        y = (cy - self.offset[1]) / self.scale
        width, height = self.display_size[0] / self.scale, self.display_size[1] / self.scale
        if not (0 <= x < width and 0 <= y < height):
            return None
        return round(x, 1), round(y, 1)

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset[0], y * self.scale + self.offset[1]

    # ---- display -----------------------------------------------------------------------

    def read_frame(self, frame: int):
        if frame != self.last_read + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.cap.read()
        self.last_read = frame if ok else -1
        return image if ok else None

    def go_to(self, frame: int) -> None:
        frame = max(0, min(frame, self.frame_count - 1))
        image = self.read_frame(frame)
        if image is None:
            self.status = f"impossible de lire la frame {frame}"
            self.draw_info()
            return
        self.current_frame = frame
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.scale != 1:
            rgb = cv2.resize(rgb, self.display_size, interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.itemconfig(self.image_id, image=self.photo)
        self.draw_annotations()
        self.draw_info()

    def draw_annotations(self) -> None:
        self.canvas.delete("ann")
        frame = self.current_frame
        event = self.ann.events.get(frame)
        position = self.ann.positions.get(frame)
        if position is not None:
            cx, cy = self.to_canvas(*position)
            self.canvas.create_line(cx, cy - 6, cx, cy + 6, width=3, fill=BALL_COLOR, tags="ann")
            self.canvas.create_line(cx - 6, cy, cx + 6, cy, width=3, fill=BALL_COLOR, tags="ann")
            if event:
                self.canvas.create_oval(
                    cx - 14,
                    cy - 14,
                    cx + 14,
                    cy + 14,
                    width=3,
                    outline=EVENT_COLORS[event],
                    tags="ann",
                )
        if event:
            label = self.canvas.create_text(
                20,
                20,
                anchor="nw",
                text=event.upper(),
                font=("Arial", 22, "bold"),
                fill=EVENT_COLORS[event],
                tags="ann",
            )
            x0, y0, x1, y1 = self.canvas.bbox(label)
            background = self.canvas.create_rectangle(
                x0 - 8, y0 - 4, x1 + 8, y1 + 4, fill="black", outline="", tags="ann"
            )
            self.canvas.tag_lower(background, label)

    def draw_info(self) -> None:
        self.info.delete("all")
        frame = self.current_frame
        event = self.ann.events.get(frame) or "-"
        counts = "  ".join(f"{e}: {n}" for e, n in self.ann.event_counts().items())
        unsaved = "  * non sauvegardé" if self.dirty else ""
        text = (
            f"frame {frame}/{self.frame_count - 1}   événement : {event}   |   {counts}"
            f"{unsaved}   {self.status}"
        )
        self.info.create_text(10, 14, anchor="w", text=text, font=("Arial", 11), fill="white")

        # Timeline of the clip: one tick per event, white cursor on the current frame
        x0, x1, y = 10, CANVAS_W - 10, 42
        self.info.create_line(x0, y, x1, y, fill="#475569", width=2)

        def to_x(f: int) -> float:
            return x0 + (x1 - x0) * f / max(self.frame_count - 1, 1)

        for f, e in self.ann.events.items():
            self.info.create_line(to_x(f), y - 8, to_x(f), y + 8, fill=EVENT_COLORS[e], width=2)
        cx = to_x(frame)
        self.info.create_polygon(cx - 5, y - 12, cx + 5, y - 12, cx, y - 4, fill="white")

    # ---- actions -----------------------------------------------------------------------

    def on_click(self, event: Event) -> None:
        position = self.to_video(event.x, event.y)
        if position is None:
            return
        self.ann.set_position(self.current_frame, *position)
        self.changed("")

    def on_key(self, event: Event) -> None:
        char = event.char or ""
        frame = self.current_frame
        if char in EVENT_KEYS:
            state = self.ann.toggle_event(frame, EVENT_KEYS[char])
            self.changed(f"{state} ajouté" if state else "événement retiré")
        elif char == "0":
            self.changed("événement retiré" if self.ann.clear_event(frame) else "")
        elif char == "d":
            self.changed("balle supprimée" if self.ann.delete_position(frame) else "")
        elif char in ("n", "p"):
            target = self.ann.next_event(frame) if char == "n" else self.ann.previous_event(frame)
            self.status = "" if target is not None else "aucun autre événement"
            if target is not None:
                self.go_to(target)
            else:
                self.draw_info()
        elif char == "s":
            self.save()
        elif char == "q":
            self.quit()

    def changed(self, status: str) -> None:
        self.dirty = True
        self.status = status
        self.draw_annotations()
        self.draw_info()

    def save(self) -> None:
        self.ann.save_csv(self.csv_path, self.frame_count)
        self.dirty = False
        self.status = f"sauvegardé dans {self.csv_path.name}"
        print(f"save annotations to {self.csv_path}")
        self.draw_info()

    def quit(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Ball Annotator", "Sauvegarder avant de quitter ?")
            if answer is None:
                return
            if answer:
                self.save()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    assert set(EVENT_KEYS.values()) == set(EVENTS)
    parser = argparse.ArgumentParser(description="Ball Annotator Tool")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video file."
    )
    args = parser.parse_args()

    root = Tk()
    BallAnnotatorApplication({"video_path": args.video_path}, root)
    root.mainloop()
