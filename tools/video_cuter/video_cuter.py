import argparse
import os
import shutil
import subprocess
import threading
from pathlib import Path
from tkinter import Canvas, DoubleVar, Menu, Scale, Tk

import cv2
from loguru import logger
from PIL import Image, ImageTk


class VideoCuterApplication:
    def __init__(self, config: dict, root: Tk) -> None:
        self.root = root
        self.video_path = config["video_path"]

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = -1
        self.pending_job: str | None = None

        self.root.title("Video Cuter Application")
        self.root.geometry("1282x820")
        self.root.resizable(False, False)
        self.video_panel = Canvas(self.root, width=1280, height=720)
        self.current_photo = ImageTk.PhotoImage(Image.new(mode="RGB", size=(1280, 720)))
        self.bg_id = self.video_panel.create_image(640, 360, image=self.current_photo)
        self.video_panel.place(x=0, y=0)

        self.position1 = DoubleVar(value=0)
        self.position2 = DoubleVar(value=0)

        self.start_scale = Scale(
            self.root,
            from_=0,
            to=self.frame_count,
            orient="horizontal",
            variable=self.position1,
            length=1280,
            command=self.seek,
        )
        self.start_scale.place(x=0, y=720)

        self.end_scale = Scale(
            self.root,
            from_=0,
            to=self.frame_count,
            orient="horizontal",
            variable=self.position2,
            length=1280,
            command=self.seek,
        )
        self.end_scale.place(x=0, y=770)

        self.active_scale = self.start_scale
        self.set_active_scale(self.start_scale)

        self.start_scale.bind("<Button-1>", lambda e: self.set_active_scale(self.start_scale), add="+")
        self.end_scale.bind("<Button-1>", lambda e: self.set_active_scale(self.end_scale), add="+")
        self.start_scale.bind("<ButtonRelease-1>", lambda e: self.root.focus_set(), add="+")
        self.end_scale.bind("<ButtonRelease-1>", lambda e: self.root.focus_set(), add="+")

        self.fps_int = max(1, int(round(self.fps)))
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Shift-Right>", lambda e: self.step(self.fps_int))
        self.root.bind("<Shift-Left>", lambda e: self.step(-self.fps_int))
        self.root.bind("<Control-Right>", lambda e: self.step(self.fps_int * 10))
        self.root.bind("<Control-Left>", lambda e: self.step(-self.fps_int * 10))
        self.root.bind("<Up>", lambda e: self.set_active_scale(self.start_scale))
        self.root.bind("<Down>", lambda e: self.set_active_scale(self.end_scale))
        self.root.bind("e", lambda e: self.export_frame())
        self.root.focus_set()

        self.menubar = Menu(self.root)
        self.file_menu = Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(
            label="New cut (fast, no re-encode)",
            command=lambda: self.cut(precise=False),
        )
        self.file_menu.add_command(
            label="New cut (precise, re-encode)",
            command=lambda: self.cut(precise=True),
        )
        self.file_menu.add_command(
            label="Export current frame as image",
            command=self.export_frame,
        )
        self.file_menu.add_command(label="Quit", command=self.close_window)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menubar)

        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix(
            "".join(Path(self.video_path).suffixes)
        )

    def show_current_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_image)
        self.current_photo = ImageTk.PhotoImage(image=img)
        self.video_panel.itemconfig(self.bg_id, image=self.current_photo)

    def set_active_scale(self, scale: Scale) -> None:
        self.active_scale = scale
        self.start_scale.config(bg="#ffd966" if scale is self.start_scale else "SystemButtonFace")
        self.end_scale.config(bg="#ffd966" if scale is self.end_scale else "SystemButtonFace")

    def step(self, delta: int) -> str:
        scale = self.active_scale
        new_val = max(0, min(self.frame_count - 1, int(scale.get()) + delta))
        scale.set(new_val)
        self.seek(new_val)
        return "break"

    def export_frame(self) -> None:
        """Save the currently displayed frame as a PNG next to the source video."""
        frame_idx = self.current_frame if self.current_frame >= 0 else int(self.active_scale.get())
        frame_idx = max(0, min(self.frame_count - 1, frame_idx))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"Could not read frame {frame_idx}")
            return

        out_path = self.directory / f"{self.video_name}_frame_{frame_idx}.png"
        if not cv2.imwrite(str(out_path), frame):
            logger.error(f"Failed to write image: {out_path}")
            return
        logger.info(f"Frame saved: {out_path}")

    def cut(self, precise: bool = False) -> None:
        start_frame = int(self.position1.get())
        end_frame = int(self.position2.get())

        if start_frame >= end_frame:
            logger.error("Start frame must be less than end frame")
            return

        ext = ".mp4" if precise else (Path(self.video_path).suffix or ".mp4")
        cut_filename = f"{self.video_name}_cut_{start_frame}_{end_frame}{ext}"
        cut_path = os.path.join(self.directory, cut_filename)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found in PATH, falling back to OpenCV")
            self.cut_opencv(start_frame, end_frame, os.path.splitext(cut_path)[0] + ".mp4")
            return

        start_t = start_frame / self.fps
        duration = (end_frame - start_frame) / self.fps

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_t:.3f}",
            "-i",
            self.video_path,
            "-t",
            f"{duration:.3f}",
        ]
        if precise:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"]
        else:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd.append(cut_path)

        threading.Thread(target=self.run_ffmpeg, args=(cmd, cut_path), daemon=True).start()

    def run_ffmpeg(self, cmd: list[str], cut_path: str) -> None:
        logger.info(f"Cutting {cut_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Cut saved: {cut_path}")
        else:
            logger.error(f"ffmpeg error:\n{result.stderr}")

    def cut_opencv(self, start_frame: int, end_frame: int, cut_path: str) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            cut_path,
            fourcc,
            self.fps,
            (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        logger.info(f"Cut saved: {cut_path}")

    def seek(self, value: str | float) -> None:
        self.current_frame = int(float(value))
        if self.pending_job is not None:
            self.root.after_cancel(self.pending_job)
        self.pending_job = self.root.after(30, self.display_pending)

    def display_pending(self) -> None:
        self.pending_job = None
        self.show_current_frame()

    def close_window(self) -> None:
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Cuter Tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    root = Tk()
    VideoCuterApplication({"video_path": args.video_path}, root)
    root.mainloop()
