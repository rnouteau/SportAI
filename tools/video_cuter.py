import argparse
import os
import threading
import cv2
from tkinter import *
from PIL import Image,ImageTk
from pathlib import Path
import subprocess
import shutil


class VideoCuterApplication:
    def __init__(self, config, root):

        self.root = root
        self.video_path = config['video_path']
        self.annotations = {}
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = -1
        self.pending_job = None

        self.root.title("Video Cuter Application")
        self.root.geometry("1282x820")
        self.root.resizable(False,False)
        self.video_panel = Canvas(self.root,width=1280,height=720)
        self.current_photo = ImageTk.PhotoImage(Image.new(mode="RGB",size=(1280,720)))
        self.bg_id = self.video_panel.create_image(640,360,image=self.current_photo)
        self.video_panel.place(x=0,y=0)

        self.position1 = DoubleVar(value=0)
        self.position2 = DoubleVar(value=0)

        self.start_scale = Scale(self.root, from_=0, to=self.frame_count, orient="horizontal",variable=self.position1, length=1280, command=self.seek)
        self.start_scale.place(x=0,y=720)

        self.end_scale = Scale(self.root, from_=0, to=self.frame_count, orient="horizontal",variable=self.position2, length=1280, command=self.seek)
        self.end_scale.place(x=0,y=770)

        self.default_bg = self.start_scale.cget("bg")
        self.active_scale = self.start_scale
        self.set_active_scale(self.start_scale)

        self.start_scale.bind("<Button-1>", lambda e: self.set_active_scale(self.start_scale),add="+")
        self.end_scale.bind("<Button-1>", lambda e: self.set_active_scale(self.end_scale),add="+")
        self.start_scale.bind("<ButtonRelease-1>", lambda e: self.root.focus_set(),add="+")
        self.end_scale.bind("<ButtonRelease-1>", lambda e: self.root.focus_set(),add="+")

        self.fps_int = max(1, int(round(self.fps)))
        self.root.bind("<Right>", lambda e: self.step(1))
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Shift-Right>", lambda e: self.step(self.fps_int))
        self.root.bind("<Shift-Left>", lambda e: self.step(-self.fps_int))
        self.root.bind("<Control-Right>", lambda e: self.step(self.fps_int*10))
        self.root.bind("<Control-Left>", lambda e: self.step(-self.fps_int*10))
        self.root.bind("<Up>",   lambda e: self.set_active_scale(self.start_scale))
        self.root.bind("<Down>", lambda e: self.set_active_scale(self.end_scale))
        self.root.focus_set()

        self.menubar = Menu(self.root)

        self.file_menu = Menu(self.menubar,tearoff=0)
        self.file_menu.add_command(label="New cut (fast, no re-encode)", command=lambda: self.cut(precise=False))
        self.file_menu.add_command(label="New cut (precise, re-encode)", command=lambda: self.cut(precise=True))
        self.file_menu.add_command(label="Quit",command=self.close_window)
        self.menubar.add_cascade(label="File",menu=self.file_menu)

        self.root.config(menu=self.menubar)

        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix("".join(Path(self.video_path).suffixes))

    def show_current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_image)
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_panel.itemconfig(self.bg_id,image=self.current_photo)

    def set_active_scale(self, scale):
        self.active_scale = scale
        self.start_scale.config(bg="#ffd966" if scale is self.start_scale else "SystemButtonFace")
        self.end_scale.config(bg="#ffd966" if scale is self.end_scale else "SystemButtonFace")

    def step(self, delta):
        scale = self.active_scale
        new_val = max(0, min(self.frame_count - 1, int(scale.get()) + delta))
        scale.set(new_val)
        self.seek(new_val)
        return "break"
    
    def cut(self, precise=False):
        start_frame = int(self.position1.get())
        end_frame = int(self.position2.get())
 
        if start_frame >= end_frame:
            print("Error: Start frame must be less than end frame.")
            return
 
        ext = ".mp4" if precise else (Path(self.video_path).suffix or ".mp4")
        cut_filename = f"{self.video_name}_cut_{start_frame}_{end_frame}{ext}"
        cut_path = os.path.join(self.directory, cut_filename)
 
        if shutil.which("ffmpeg") is None:
            print("No ffmpeg  in the PATH : fallback OpenCV.")
            self.cut_opencv(start_frame, end_frame, os.path.splitext(cut_path)[0] + ".mp4")
            return
 
        start_t = start_frame / self.fps
        duration = (end_frame - start_frame) / self.fps
 
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start_t:.3f}", "-i", self.video_path, "-t", f"{duration:.3f}"]
        if precise:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"]
        else:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd.append(cut_path)

        threading.Thread(target=self.run_ffmpeg, args=(cmd, cut_path), daemon=True).start()

    def run_ffmpeg(self, cmd, cut_path):
        print(f"Cutting... {cut_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Cut saved: {cut_path}")
        else:
            print(f"ffmpeg error:\n{result.stderr}")
 
    def cut_opencv(self, start_frame, end_frame, cut_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cut_path, fourcc, self.fps,
                              (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)
 
        out.release()
        print(f"Cut saved: {cut_path}")

    def seek(self, value):
        self.current_frame = int(float(value))
        if self.pending_job is not None:
            self.root.after_cancel(self.pending_job)
        self.pending_job = self.root.after(30, self.display_pending)
 
    def display_pending(self):
        self.pending_job = None
        self.show_current_frame()

    def close_window(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Cuter Tool")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    video_path = args.video_path
    config = {
        'video_path': video_path,
    }

    print(config)

    root = Tk()
    app = VideoCuterApplication(config,root)
    root.mainloop()


    



