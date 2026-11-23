"""Draw tennis analysis predictions onto a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.objects.tennis.tennis_analysis import TennisAnalysis
from pipeline.objects.tennis.tennis_court_schema import TennisCourtSchema
from pipeline.video_reader import VideoReader

BALL_COLOR = (0, 255, 255)
PLAYER_COLOR = (255, 128, 0)
LEGS_COLOR = (0, 215, 255)
COURT_COLOR = (0, 255, 0)
KEYPOINT_COLOR = (255, 0, 255)
# One colour per track id (BGR), so an id switch is visible at a glance.
ID_PALETTE = (
    (255, 128, 0),
    (0, 128, 255),
    (60, 200, 60),
    (200, 60, 200),
    (0, 200, 200),
    (60, 60, 255),
    (255, 200, 120),
    (120, 255, 200),
)


def player_color(track_id: int | None) -> tuple[int, int, int]:
    """Colour of a player box: fixed per track id, default colour without id."""
    if track_id is None:
        return PLAYER_COLOR
    return ID_PALETTE[track_id % len(ID_PALETTE)]


def _player_label(player_id: int | None, track_id: int | None) -> str:
    """``"P1 #7"`` (selected player + tracker id), ``"ID 7"`` or ``""``."""
    if player_id is not None:
        return f"P{player_id} #{track_id}" if track_id is not None else f"P{player_id}"
    return f"ID {track_id}" if track_id is not None else ""


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Text on a filled box whose bottom-left corner is (x, y); kept inside the image."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - w - 4))
    y = max(h + baseline + 2, y)
    cv2.rectangle(frame, (x, y - h - baseline - 2), (x + w + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, analysis_frame) -> None:
    ball = analysis_frame.ball
    if ball is None or len(ball.position) < 2:
        return
    x, y = int(round(ball.position[0])), int(round(ball.position[1]))
    cv2.circle(frame, (x, y), 6, BALL_COLOR, 2, cv2.LINE_AA)
    conf = ball.confidence
    if conf is not None:
        cv2.putText(
            frame,
            f"ball {conf:.2f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            BALL_COLOR,
            1,
            cv2.LINE_AA,
        )


def _draw_players(frame: np.ndarray, analysis_frame) -> None:
    for player in analysis_frame.players:
        track_id, player_id = player.track_id, player.player_id
        # Colour follows the stable player id when the selector ran, else the tracker id.
        color = player_color(player_id if player_id is not None else track_id)
        bbox = player.bbox
        if bbox is not None and len(bbox) >= 4:
            x0, y0, x1, y1 = (int(round(v)) for v in bbox[:4])
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            label = _player_label(player_id, track_id)
            if label:
                _draw_label(frame, label, x0, y0 - 2, color)

        image_pt = player.feet
        if image_pt is not None:
            px, py = int(round(image_pt.x)), int(round(image_pt.y))
            cv2.circle(frame, (px, py), 4, LEGS_COLOR, -1, cv2.LINE_AA)
            court = player.court
            if court is not None:
                cv2.putText(
                    frame,
                    f"d={court.distance:.1f}m",
                    (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    LEGS_COLOR,
                    1,
                    cv2.LINE_AA,
                )


def _draw_court(frame: np.ndarray, analysis_frame, analysis: TennisAnalysis) -> None:
    homo = analysis_frame.homography
    if homo is None:
        return

    schema = analysis.court_schema
    if homo.matrix is not None and isinstance(schema, TennisCourtSchema):
        h_img_to_court = np.asarray(homo.matrix, dtype=np.float64)
        try:
            h_court_to_img = np.linalg.inv(h_img_to_court)
        except np.linalg.LinAlgError:
            logger.warning("Homography matrix is not invertible; skipping court lines")
            return
        for (x0, y0), (x1, y1) in schema.overlay_segments_m():
            pts = np.array([[[x0, y0], [x1, y1]]], dtype=np.float64)
            proj = cv2.perspectiveTransform(pts, h_court_to_img).reshape(-1, 2)
            p0 = (int(round(proj[0, 0])), int(round(proj[0, 1])))
            p1 = (int(round(proj[1, 0])), int(round(proj[1, 1])))
            cv2.line(frame, p0, p1, COURT_COLOR, 2, cv2.LINE_AA)
        return

    keypoints = homo.keypoints
    if not keypoints:
        return
    for pt in keypoints:
        if len(pt) < 2:
            continue
        cv2.drawMarker(
            frame,
            (int(round(pt[0])), int(round(pt[1]))),
            KEYPOINT_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )


def render_overlay_video(
    video_reader: VideoReader,
    analysis: TennisAnalysis,
    output_path: str | Path,
) -> None:
    """Write an annotated video next to the analysis predictions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fps = float(video_reader.fps) if video_reader.fps and video_reader.fps > 0 else 25.0
    size = (int(video_reader.width), int(video_reader.height))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out}")

    n_frames = min(len(analysis.frames), video_reader.frame_count)
    logger.info(f"Writing overlay video to {out} ({n_frames} frames)")

    for frame_id, image in video_reader.iter_frames():
        if frame_id >= n_frames:
            break
        canvas = image.copy()
        analysis_frame = analysis.frames[frame_id]
        _draw_ball(canvas, analysis_frame)
        _draw_players(canvas, analysis_frame)
        _draw_court(canvas, analysis_frame, analysis)
        writer.write(canvas)

    writer.release()
    logger.info(f"Overlay video saved: {out}")
