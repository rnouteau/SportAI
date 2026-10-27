import argparse
import os
import cv2
from tkinter import *
import json
from PIL import Image,ImageTk
import csv
import pandas as pd
from pathlib import Path


class Cut():
    def __init__(self,canvas : Canvas,min_val,max_val,height,width,min_gap,command):

        self.canvas = canvas
        self.min_val,self.max_val = min_val,max_val
        self.start,self.end = min_val,max_val
        self.height,self.width = height,width
        self.min_gap = min_gap
        self.command = command
        self.r = 8
        self.id = 0



    def draw(self):
        self.canvas.create_oval(self.min_val - self.r,self.height - self.r, self.min_val + self.r, self.height + self.r, fill="blue")
        self.canvas.create_oval(self.min_val - self.r + self.min_gap,self.height - self.r, self.min_val + self.r + self.min_gap, self.height + self.r, fill="blue")



class VideoCuterApplication:
    def __init__(self, config, root):

        self.root = root
        self.video_path = config['video_path']
        self.annotations = {}
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = -1

        self.root.title("Video Cuter Application")
        self.root.geometry("1282x820")
        self.root.resizable(False,False)
        self.video_panel = Canvas(self.root,width=1280,height=720)
        self.current_photo = ImageTk.PhotoImage(Image.new(mode="RGB",size=(1280,720)))
        self.bg_id = self.video_panel.create_image(640,360,image=self.current_photo)
        self.video_panel.place(x=0,y=0)
        # self.position = DoubleVar(value=0)
        # self.lecture_bar = Scale(self.root,from_=0,to=self.frame_count,orient="horizontal",variable=self.position,length=1280,command=self.seek)
        # self.lecture_bar.place(x=0,y=720)
        
        self.menubar = Menu(self.root)

        self.file_menu = Menu(self.menubar,tearoff=0)
        self.file_menu.add_command(label="New cut",command=self.create_new_cut)
        self.file_menu.add_command(label="Delete last cut",command=self.delete_last_cut)
        self.file_menu.add_command(label="Save",command=self.save)
        self.file_menu.add_command(label="Quit",command=self.close_window)
        self.menubar.add_cascade(label="File",menu=self.file_menu)

        self.root.config(menu=self.menubar)

        self.range_slider = Canvas(self.root,width=1280,height=50,highlightthickness=0)
        # self.range_slider.create_text(30,40,text="Hello, Tkinter!", font=("Arial", 24), fill="blue")
        self.range_slider.create_rectangle(10,20,1270,30)
        self.range_slider.place(x=0,y=720)


        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix("".join(Path(self.video_path).suffixes))

        self.cuts = []


    def show_current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2_image_resized = cv2.resize()
            img = Image.fromarray(cv2_image)
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_panel.itemconfig(self.bg_id,image=self.current_photo)

    # def seek(self,value):
    #     self.current_frame = int(value)
    #     self.show_current_frame()

    def close_window(self):
        self.root.destroy()

    def save(self):
        pass

    def create_new_cut(self):
        self.cut = Cut(self.range_slider,10,1270,25,30,30,None) 
        self.cut.draw()
        self.cuts.append(self.cut)

    def delete_last_cut(self):
        pass

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


    



