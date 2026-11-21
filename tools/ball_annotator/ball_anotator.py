import argparse
import os
from pathlib import Path
from tkinter import Canvas, Event, Tk

import cv2
import pandas as pd
from PIL import Image, ImageTk


class BallAnnotatorApplication:
    def __init__(self, config: dict, root: Tk) -> None:
        self.root = root
        self.video_path = config["video_path"]
        self.annotations: dict[int, tuple[float, float]] = {}

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = -1

        self.root.title("Ball Annotator Tool")
        self.root.geometry("1282x820")
        self.root.resizable(False, False)
        self.video_panel = Canvas(self.root, width=1280, height=720)
        self.current_photo = ImageTk.PhotoImage(Image.new(mode="RGB", size=(1280, 720)))
        self.bg_id = self.video_panel.create_image(640, 360, image=self.current_photo)
        self.video_panel.place(x=0, y=0)
        self.info_panel = Canvas(self.root, bg="black", width=1280, height=50)
        self.text_frame_id = self.info_panel.create_text(
            50,
            25,
            text=f"frame : {self.current_frame}",
            font=("Arial", 12),
            fill="white",
        )
        self.info_panel.place(x=0, y=720)
        self.video_panel.pack()

        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>", self.previous_frame)
        self.root.bind("q", self.close_window)
        self.root.bind("d", self.delete)
        self.root.bind("s", self.export_annotations)
        self.video_panel.bind("<Button-1>", self.paint)

        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix(
            "".join(Path(self.video_path).suffixes)
        )
        self.import_annotations()

    def show_current_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_image)
        self.current_photo = ImageTk.PhotoImage(image=img)
        self.video_panel.itemconfig(self.bg_id, image=self.current_photo)
        self.info_panel.itemconfig(self.text_frame_id, text=f"frame : {self.current_frame}")

        self.video_panel.delete("line_tag")
        if self.current_frame in self.annotations:
            x, y = self.annotations[self.current_frame]
            self.video_panel.create_line(x, y - 5, x, y + 5, width=3, fill="blue", tags=("line_tag",))
            self.video_panel.create_line(x - 5, y, x + 5, y, width=3, fill="blue", tags=("line_tag",))

    def next_frame(self, event: Event) -> None:
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            self.show_current_frame()

    def previous_frame(self, event: Event) -> None:
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_current_frame()

    def close_window(self, event: Event) -> None:
        self.cap.release()
        self.root.destroy()

    def paint(self, event: Event) -> None:
        if self.current_frame in self.annotations:
            self.video_panel.delete("line_tag")

        self.annotations[self.current_frame] = (event.x, event.y)
        self.video_panel.create_line(
            event.x, event.y - 5, event.x, event.y + 5, width=3, fill="blue", tags=("line_tag",)
        )
        self.video_panel.create_line(
            event.x - 5, event.y, event.x + 5, event.y, width=3, fill="blue", tags=("line_tag",)
        )

    def delete(self, event: Event) -> None:
        self.annotations.pop(self.current_frame, None)
        self.video_panel.delete("line_tag")

    def import_annotations(self) -> None:
        annotations_path = f"{self.directory}/{self.video_name}.csv"
        if not os.path.exists(annotations_path):
            return
        print(f"import annotations from {annotations_path}")
        data = pd.read_csv(annotations_path).dropna(subset=["x", "y"])
        for row in data.itertuples(index=False):
            self.annotations[int(row.frame)] = (row.x, row.y)


    def export_annotations(self, event: Event) -> None:
        df = pd.DataFrame.from_dict(self.annotations, orient="index", columns=["x", "y"])
        df = df.reindex(range(self.frame_count))
        path = f"{self.directory}/{self.video_name}.csv"
        df.to_csv(path, index_label="frame")
        print(f"save annotations to{path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball Annotator Tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    root = Tk()
    BallAnnotatorApplication({"video_path": args.video_path}, root)
    root.mainloop()
