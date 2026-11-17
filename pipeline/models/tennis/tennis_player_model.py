
import torch

from pipeline.models.model import Model
from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.video_reader import VideoReader
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from ultralytics import YOLO


class TennisPlayerModel(Model):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisPlayerModel initialized")
        self.weight = self.config.get("weight", None)
        self.device = self.config.get("device", "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU instead")
            self.device = "cpu"
        self.model = YOLO(self.weight)
        self.input_size = self.config.get("input_size", (640, 640))

        if self.model:
            logger.info(f"Model loaded with weight: {self.weight} on device: {self.device}")

    def run(self, video_reader : VideoReader, analysis : TennisAnalysis) -> None:
        logger.info("Running TennisPlayerModel")
        if self.model is None:
            logger.error("No weight loaded, cannot run")
            return
        for frame in analysis.frames:
            image = video_reader.read_frame(frame.id)
            results = self.model.predict(
                        image,
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
                analysis.frames[frame.id].players = [
                    TennisPlayerAtFrame.create(xyxy[i].tolist(), float(confs[i])) for i in top
                ]
