
import cv2

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        # Initialize other necessary attributes here

        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def read_frame(self, frame_number: int) -> any:
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} is out of bounds for video with {self.frame_count} frames.")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video.")
        
        return frame

    def info(self) -> dict:
        return {
            "video_path": self.video_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "height": self.height,
            "width": self.width
        }

    def release(self) -> None:
        self.cap.release()
