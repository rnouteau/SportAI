from loguru import logger
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch

from pipeline.device import resolve_device
from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.video_reader import VideoReader


class TennisPlayerModel(Model[TennisAnalysis]):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisPlayerModel initialized")
        weight = self.config.get("weight")
        self.weight: str | None = str(weight) if weight else None
        self.device = resolve_device(str(self.config.get("device", "cpu")))
        self.model: YOLO | None = YOLO(self.weight) if self.weight else None
        self.input_size = self.config.get("input_size", (640, 640))

        if self.model:
            logger.info(f"Model loaded with weight: {self.weight} on device: {self.device}")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisPlayerModel")
        if self.model is None:
            logger.error("No weight loaded, cannot run")
            return

        n_frames = len(analysis.frames)
        for frame_id, image in video_reader.iter_frames():
            if frame_id >= n_frames:
                break
            results = self.model.predict(
                image,
                classes=[0],
                stream=True,
                verbose=False,
                device=self.device,
            )
            for result in results:
                if not isinstance(result, Results):
                    continue
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                if boxes.conf is None or boxes.xyxy is None:
                    continue
                confs = torch.as_tensor(boxes.conf).detach().cpu().numpy()
                xyxy = torch.as_tensor(boxes.xyxy).detach().cpu().numpy()
                top = confs.argsort()[::-1][:2]
                analysis.frames[frame_id].players = [
                    TennisPlayerAtFrame.create(xyxy[i].tolist(), float(confs[i])) for i in top
                ]
