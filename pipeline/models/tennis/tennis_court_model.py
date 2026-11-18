from loguru import logger
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pipeline.device import resolve_device
from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.video_reader import VideoReader


class TennisCourtModel(Model[TennisAnalysis]):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisCourtModel initialized")
        weight = self.config.get("weight")
        self.weight: str | None = str(weight) if weight else None
        self.device = resolve_device(str(self.config.get("device", "cpu")))
        self.model: YOLO | None = YOLO(self.weight) if self.weight else None
        self.input_size = self.config.get("input_size", (640, 640))

        if self.model:
            logger.info(f"Model loaded with weight: {self.weight} on device: {self.device}")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisCourtModel")

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
                keypoints = result.keypoints
                if keypoints is None or len(keypoints) == 0:
                    continue
                if keypoints.conf is None or keypoints.xy is None:
                    continue

                # Shapes: xy (N, K, 2), conf (N, K) — keep best instance.
                xy = torch.as_tensor(keypoints.xy).detach().cpu()
                conf = torch.as_tensor(keypoints.conf).detach().cpu()
                if xy.ndim != 3 or conf.ndim != 2 or xy.shape[0] == 0:
                    continue
                best = int(conf.mean(dim=1).argmax().item())
                pts = xy[best].tolist()  # list[[x, y], ...]
                confs = conf[best].tolist()  # list[float]
                schema = analysis.court_schema
                if schema is None:
                    logger.error("No court_schema on analysis, cannot estimate homography")
                    continue
                h_mat = schema.estimate_homography(pts, confs)
                homo = TennisHomography.create(keypoints=pts, confidences=confs)
                if h_mat is not None:
                    homo.matrix = h_mat.tolist()
                analysis.frames[frame_id].homography = homo
