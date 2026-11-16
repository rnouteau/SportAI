from collections.abc import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def read_frame(self, frame_number: int) -> np.ndarray:
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(
                f"Frame number {frame_number} is out of bounds "
                f"for video with {self.frame_count} frames."
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video.")

        return frame

    def _read_next(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError("Could not read the next frame from video.")
        # Copy: some OpenCV backends reuse the decode buffer across reads.
        return frame.copy()

    def iter_triplets(self) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (index, prev, curr, next) with edge clamping, in one forward pass.

        Same semantics as:
          prev = read_frame(max(i - 1, 0))
          curr = read_frame(i)
          next = read_frame(min(i + 1, n - 1))
        """
        n = self.frame_count
        if n <= 0:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if n == 1:
            frame = self._read_next()
            yield 0, frame, frame, frame
            return

        frame_0 = self._read_next()
        frame_1 = self._read_next()
        yield 0, frame_0, frame_0, frame_1

        prev, curr = frame_0, frame_1
        for index in range(1, n - 1):
            nxt = self._read_next()
            yield index, prev, curr, nxt
            prev, curr = curr, nxt

        yield n - 1, prev, curr, curr

    def info(self) -> dict:
        return {
            "video_path": self.video_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "height": self.height,
            "width": self.width,
        }

    def release(self) -> None:
        self.cap.release()
