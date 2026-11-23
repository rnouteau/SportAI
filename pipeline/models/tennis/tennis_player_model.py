import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pipeline.device import resolve_device
from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame
from pipeline.video_reader import VideoReader

DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.1


def _to_numpy(values: object) -> np.ndarray:
    return torch.as_tensor(values).detach().cpu().numpy()


class TennisPlayerModel(Model[TennisAnalysis]):
    """Player detection + tracking (YOLO + ByteTrack).

    Config keys:
      weight, device
      tracker: "bytetrack.yaml"   # Ultralytics tracker yaml (built-in name or path)
      conf: 0.1                   # detection threshold; keep it <= track_low_thresh
      max_players: null           # keep the N most confident people per frame (null = all);
                                  # keep everyone when TennisPlayerSelectorStat runs after
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisPlayerModel initialized")
        weight = self.config.get("weight")
        self.weight: str | None = str(weight) if weight else None
        self.device = resolve_device(str(self.config.get("device", "cpu")))
        self.model: YOLO | None = YOLO(self.weight) if self.weight else None
        self.tracker = str(self.config.get("tracker") or DEFAULT_TRACKER)
        conf = self.config.get("conf")
        self.conf = float(conf) if conf is not None else DEFAULT_CONF
        self.input_size = self.config.get("input_size", (640, 640))
        max_players = self.config.get("max_players")
        self.max_players: int | None = int(max_players) if max_players is not None else None

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
            # persist=True keeps the tracker state between frames (one model per video).
            results = self.model.track(
                image,
                classes=[0],
                stream=True,
                verbose=False,
                device=self.device,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
            )
            for result in results:
                if not isinstance(result, Results):
                    continue
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                if boxes.conf is None or boxes.xyxy is None:
                    continue
                confs = _to_numpy(boxes.conf)
                xyxy = _to_numpy(boxes.xyxy)
                # No id when no detection was confident enough to start a track
                # (score < new_track_thresh): keep the boxes, without track_id.
                ids = [int(v) for v in _to_numpy(boxes.id)] if boxes.id is not None else None
                players = []
                for i in confs.argsort()[::-1][: self.max_players]:
                    player = TennisPlayerAtFrame(bbox=xyxy[i].tolist(),confidence=float(confs[i]),track_id=ids[i] if ids is not None else None)
                    players.append(player)
                analysis.frames[frame_id].players = players
