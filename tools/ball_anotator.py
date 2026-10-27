import argparse
import os
import cv2
from tkinter import *
import json
from PIL import Image,ImageTk
import csv
import pandas as pd
from pathlib import Path

class BallAnnotatorApplication:
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

        self.root.title("Ball Annotator Tool")
        self.root.geometry("1282x820")
        self.root.resizable(False,False)
        self.video_panel = Canvas(self.root,width=1280,height=720)
        self.current_photo = ImageTk.PhotoImage(Image.new(mode="RGB",size=(1280,720)))
        self.bg_id = self.video_panel.create_image(640,360,image=self.current_photo)
        self.video_panel.place(x=0,y=0)
        self.info_panel = Canvas(self.root,bg="black",width=1280,height=50)
        self.text_frame_id = self.info_panel.create_text(50, 25, text=f"frame : {self.current_frame}", font=("Arial", 12), fill="white")
        self.info_panel.place(x=0,y=720)
        
        self.video_panel.pack()

        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>",self.previous_frame)
        self.root.bind("q",self.close_window)
        self.root.bind("d",self.delete)
        self.root.bind("s",self.export_annotations)
        self.video_panel.bind("<Button-1>",self.paint)

        self.directory = Path(self.video_path).parent.absolute()
        self.video_name = Path(self.video_path).name.removesuffix("".join(Path(self.video_path).suffixes))

        self.import_annotations()


    def show_current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2_image_resized = cv2.resize()
            img = Image.fromarray(cv2_image)
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_panel.itemconfig(self.bg_id,image=self.current_photo)
            self.info_panel.itemconfig(self.text_frame_id,text=f"frame : {self.current_frame}")

            self.video_panel.delete('line_tag')
            if self.current_frame in self.annotations.keys():
                x,y = self.annotations[self.current_frame]
                self.video_panel.create_line(x,y - 5,x,y + 5 ,width=3,fill="blue",tags=('line_tag'))
                self.video_panel.create_line(x - 5,y,x + 5,y,width=3,fill="blue",tags=('line_tag'))


    def next_frame(self,event):
        if self.current_frame < self.frame_count:
            self.current_frame += 1
            self.show_current_frame()
        else:
            print("pas possible")

    def previous_frame(self,event):
        print(self.current_frame)
        if self.current_frame > 0:
            self.current_frame -= 1
            print(self.current_frame)
            self.show_current_frame()
        else:
            print("pas possible")

    def close_window(self,event):
        self.root.destroy()

    def paint(self,event):
        if self.current_frame in self.annotations.keys():
            self.video_panel.delete('line_tag')

        self.annotations[self.current_frame] = (event.x,event.y)
        self.video_panel.create_line(event.x,event.y - 5,event.x,event.y + 5 ,width=3,fill="blue",tags=('line_tag'))
        self.video_panel.create_line(event.x - 5,event.y,event.x + 5,event.y,width=3,fill="blue",tags=('line_tag'))

    def delete(self,event):
        self.annotations.pop(self.current_frame)
        self.video_panel.delete('line_tag')

    def import_annotations(self):
        data = pd.read_csv(f'{self.directory}/{self.video_name}.csv', header=None, skiprows=1)
        for row in data.iloc():
            self.annotations[row[0]] = (row[1],row[2])

        print(self.annotations)

    def export_annotations(self,event):
        print(self.annotations)
        pd.DataFrame.from_dict(self.annotations, orient='index', columns=["x", "y"]).to_csv(f'{self.directory}/{self.video_name}.csv', index_label="frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball Annotator Tool")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()

    video_path = args.video_path
    config = {
        'video_path': video_path,
    }

    print(config)

    root = Tk()
    app = BallAnnotatorApplication(config,root)
    root.mainloop()


    



