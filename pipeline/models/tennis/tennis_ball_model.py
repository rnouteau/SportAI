from loguru import logger
import numpy as np
import torch
import cv2

from pipeline.models.model import Model
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.video_reader import VideoReader


class TennisBallModel(Model):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("TennisBallModel initialized")
        self.weight = self.config.get("weight", None)
        self.device = self.config.get("device", "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is not available, using CPU instead")
            self.device = "cpu"
        self.model = (
            torch.jit.load(self.weight, map_location=self.device).eval()
            if self.weight
            else None
        )
        self.threshold = self.config.get("threshold", 0.5)
        self.input_size = self.config.get("input_size", (288, 512))

        if self.model:
            logger.info(f"Model loaded with weight: {self.weight} on device: {self.device}")

    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        logger.info("Running TennisBallModel")
        if self.model is None:
            logger.error("No weight loaded, cannot run")
            return

        n = video_reader.frame_count
        in_h, in_w = self.input_size
        threshold_uint8 = int(self.threshold * 255)

        for frame_number in range(n):
            prev_idx = max(frame_number - 1, 0)
            next_idx = min(frame_number + 1, n - 1)

            frame_prev = video_reader.read_frame(prev_idx)
            frame = video_reader.read_frame(frame_number)
            frame_next = video_reader.read_frame(next_idx)

            if frame is None or frame_prev is None or frame_next is None:
                logger.warning(f"Missing frame around {frame_number}, skipped")
                continue

            input_tensor = self._preprocess_frame(frame, frame_prev, frame_next)

            with torch.no_grad():
                heatmap_preds = self.model(input_tensor)

            heatmaps_np = heatmap_preds.squeeze(0).cpu().numpy()
            heatmap_uint8 = (heatmaps_np[1] * 255).astype(np.uint8)

            coords = self._heatmap_to_coords(heatmap_uint8, threshold=threshold_uint8)

            if coords is not None:
                h, w = frame.shape[:2]
                coords = (int(coords[0] * w / in_w), int(coords[1] * h / in_h))
                analysis.frames[frame_number].ball = TennisBall.create(
                    position=coords,
                    confidence=float(heatmaps_np[1].max()),
                )

    def _resize(self, keys: list[str], results: dict) -> dict:
        for key in keys:
            if key in results:
                results[key] = cv2.resize(results[key], (self.input_size[1], self.input_size[0]))
        return results

    def _concatenate(self, keys: list[str], results: dict, output_key: str = "image") -> dict:
        imgs_to_stack = [results[key] for key in keys]
        results[output_key] = np.concatenate(imgs_to_stack, axis=2)
        return results

    def _preprocess_frame(
        self,
        frame: np.ndarray,
        frame_prev: np.ndarray,
        frame_next: np.ndarray,
    ) -> torch.Tensor:
        frame1_rgb = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)

        data_dict = {"path_prev": frame1_rgb, "path": frame2_rgb, "path_next": frame3_rgb}
        data_dict = self._resize(["path_prev", "path", "path_next"], data_dict)
        data_dict = self._concatenate(["path_prev", "path", "path_next"], data_dict)

        image_np = data_dict["image"]
        return (
            torch.from_numpy(image_np.transpose(2, 0, 1))
            .float()
            .div(255)
            .unsqueeze(0)
            .to(self.device)
        )

    def _heatmap_to_coords(
        self,
        heatmap: np.ndarray,
        threshold: int = 127,
    ) -> tuple[int, int] | None:
        if heatmap.dtype != np.uint8:
            heatmap = heatmap.astype(np.uint8)

        _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return cx, cy

        return None
