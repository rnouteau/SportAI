import argparse
import os
from pathlib import Path
from tkinter import Canvas, Event, Tk

import cv2
import pandas as pd
from PIL import Image, ImageTk
from ultralytics import YOLO


class BallAnnotatorApplication:
    def __init__(self, config: dict, root: Tk) -> None:
        self.root = root
        self.video_path = config["video_path"]
        self.annotations: dict = {}

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = -1

        self.root.title("PLayer Annotator Tool")
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
        self.video_panel.bind("<ButtonRelease-1>", self.paint_two)
        self.video_panel.bind("<Motion>", self._track_mouse)

        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix(
            "".join(Path(self.video_path).suffixes)
        )

        self.pos = []
        self.mouse_x = 0
        self.mouse_y = 0
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
        self.video_panel.delete("rect_tag")
        if self.current_frame in self.annotations and self.annotations[self.current_frame] is not None:
            for bbox in self.annotations[self.current_frame]:
                _,x0,y0,x1,y1 = bbox
                self.video_panel.create_rectangle(
                            x0, y0, x1, y1, width=1, tags=("rect_tag",)
                        )

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
        self.pos.append(event.x)
        self.pos.append(event.y)

    def paint_two(self,event:Event) -> None:
        self.video_panel.create_rectangle(
                                    self.pos[0], self.pos[1],event.x,event.y, width=1, tags=("rect_tag",)
                                )

        if self.current_frame not in self.annotations.keys():
            self.annotations[self.current_frame] = [(0, self.pos[0],self.pos[1],event.x,event.y)]
        else:
            self.annotations[self.current_frame].append((0, self.pos[0],self.pos[1],event.x,event.y))

        self.pos = []


    def _track_mouse(self, event: Event) -> None:
        self.mouse_x = event.x
        self.mouse_y = event.y

    def delete(self, event: Event) -> None:
        bboxes = self.annotations.get(self.current_frame)
        if not bboxes:
            return
        hit_idx = None
        for i, (_, x0, y0, x1, y1) in enumerate(bboxes):
            xa, xb = min(x0, x1), max(x0, x1)
            ya, yb = min(y0, y1), max(y0, y1)
            if xa <= self.mouse_x <= xb and ya <= self.mouse_y <= yb:
                hit_idx = i
        if hit_idx is None:
            return
        bboxes.pop(hit_idx)
        if not bboxes:
            self.annotations.pop(self.current_frame, None)
        self.show_current_frame()

    def import_annotations(self) -> None:
        annotations_path = f"{self.directory}/{self.video_name}_players.csv"
        if not os.path.exists(annotations_path):
            print(f"pre-annotate video {annotations_path}")
            self.pre_annotate()
            return
        print(f"import annotations from {annotations_path}")
        data = pd.read_csv(annotations_path).dropna(subset=["class_id","x0", "y0","x1","y1"])
        for row in data.itertuples(index=False):
            if int(row.frame) not in self.annotations.keys():
                self.annotations[int(row.frame)] = [(row.class_id, row.x0,row.y0,row.x1,row.y1)]
            else:
                self.annotations[int(row.frame)].append((row.class_id, row.x0,row.y0,row.x1,row.y1))

    def pre_annotate(self) -> None:
        weights = Path(__file__).resolve().parents[1] / "models" / "based_models" / "yolo11s.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"Missing pretrained weights at {weights}. "
                "Place yolo11s.pt under models/based_models/."
            )
        model = YOLO(str(weights))
        results = model.predict(
            self.video_path,
            classes=[0],
            stream=True,
            verbose=False,
        )
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            top = confs.argsort()[::-1][:2]
            self.annotations[frame_idx] = [
                (0, float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3]))
                for i in top
            ]

    def _save_annotations_csv(self) -> None:
        rows = []
        for frame, bboxes in self.annotations.items():
            for class_id, x0, y0, x1, y1 in bboxes:
                rows.append(
                    {"frame": frame, "class_id": class_id, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
                )
        df = pd.DataFrame(rows)
        path = f"{self.directory}/{self.video_name}_players.csv"
        df.to_csv(path, index=False)
        print(f"save annotations to {path}")

    def export_annotations(self, event: Event) -> None:
        self._save_annotations_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball Annotator Tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    root = Tk()
    BallAnnotatorApplication({"video_path": args.video_path}, root)
    root.mainloop()
