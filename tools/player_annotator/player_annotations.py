"""Player track annotations: data model, edits, CSV and MOT I/O (no GUI, no model).

Used by ``tools/player_annotator.py``. Kept free of Tk / Ultralytics so it can be
unit-tested.

CSV (``<video>_players.csv``, read by the player detection evaluator):
    frame, class_id, track_id, x0, y0, x1, y1, keyframe, reviewed
- coordinates are in original video pixels, x0 <= x1 and y0 <= y1;
- a frame checked by a human but without any player is stored as one row with
  empty coordinates and ``reviewed = 1`` (the evaluator drops it with ``dropna``);
- legacy CSVs (no ``track_id`` column) load with ``track_id = -1`` and every frame
  that has boxes counts as reviewed.

MOT export (``<video>_mot/gt/gt.txt``): ``frame(1-based), id, x, y, w, h, 1, 1, 1.0``,
only for reviewed frames and boxes with a known id.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

UNKNOWN_ID = -1
CSV_COLUMNS = ["frame", "class_id", "track_id", "x0", "y0", "x1", "y1", "keyframe", "reviewed"]


@dataclass(frozen=True)
class PlayerBox:
    track_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    class_id: int = 0
    keyframe: bool = False

    @classmethod
    def from_corners(
        cls,
        track_id: int,
        xa: float,
        ya: float,
        xb: float,
        yb: float,
        *,
        class_id: int = 0,
        keyframe: bool = False,
    ) -> PlayerBox:
        """Build a box from two opposite corners in any order."""
        return cls(
            track_id=int(track_id),
            x0=float(min(xa, xb)),
            y0=float(min(ya, yb)),
            x1=float(max(xa, xb)),
            y1=float(max(ya, yb)),
            class_id=int(class_id),
            keyframe=bool(keyframe),
        )

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


def _lerp_box(a: PlayerBox, b: PlayerBox, t: float) -> PlayerBox:
    return PlayerBox(
        track_id=b.track_id,
        x0=a.x0 + (b.x0 - a.x0) * t,
        y0=a.y0 + (b.y0 - a.y0) * t,
        x1=a.x1 + (b.x1 - a.x1) * t,
        y1=a.y1 + (b.y1 - a.y1) * t,
        class_id=b.class_id,
        keyframe=False,
    )


class PlayerAnnotations:
    """Boxes per frame (one box per known track id and frame) + reviewed frames."""

    def __init__(self) -> None:
        self.boxes: dict[int, list[PlayerBox]] = {}
        self.reviewed: set[int] = set()

    # ------------------------------------------------------------------ queries
    def frame_boxes(self, frame: int) -> list[PlayerBox]:
        return list(self.boxes.get(frame, []))

    def is_reviewed(self, frame: int) -> bool:
        return frame in self.reviewed

    def box_at(self, frame: int, x: float, y: float) -> PlayerBox | None:
        """Smallest box containing (x, y), so a nested box stays selectable."""
        hits = [b for b in self.boxes.get(frame, []) if b.contains(x, y)]
        return min(hits, key=lambda b: b.area) if hits else None

    def track_ids(self) -> list[int]:
        return sorted({b.track_id for boxes in self.boxes.values() for b in boxes})

    def previous_keyframe(self, frame: int, track_id: int) -> tuple[int, PlayerBox] | None:
        for f in sorted((f for f in self.boxes if f < frame), reverse=True):
            for b in self.boxes[f]:
                if b.track_id == track_id and b.keyframe:
                    return f, b
        return None

    def unreviewed_frames(self, frame_count: int) -> list[int]:
        return [f for f in range(frame_count) if f not in self.reviewed]

    # ------------------------------------------------------------------ edits
    def _store(self, frame: int, boxes: list[PlayerBox]) -> None:
        if boxes:
            self.boxes[frame] = boxes
        else:
            self.boxes.pop(frame, None)

    def mark_reviewed(self, frame: int) -> None:
        self.reviewed.add(frame)

    def toggle_reviewed(self, frame: int) -> bool:
        if frame in self.reviewed:
            self.reviewed.discard(frame)
            return False
        self.reviewed.add(frame)
        return True

    def set_box(self, frame: int, box: PlayerBox, *, review: bool = True) -> None:
        """Add a box; it replaces the box with the same known id in that frame."""
        boxes = self.boxes.get(frame, [])
        if box.track_id != UNKNOWN_ID:
            boxes = [b for b in boxes if b.track_id != box.track_id]
        self._store(frame, [*boxes, box])
        if review:
            self.mark_reviewed(frame)

    def delete_at(self, frame: int, x: float, y: float) -> bool:
        target = self.box_at(frame, x, y)
        if target is None:
            return False
        boxes = list(self.boxes[frame])
        boxes.remove(target)
        self._store(frame, boxes)
        self.mark_reviewed(frame)
        return True

    def reassign_at(self, frame: int, x: float, y: float, new_id: int) -> bool:
        """Give ``new_id`` to the box under (x, y); a box already holding it takes the old id."""
        target = self.box_at(frame, x, y)
        if target is None:
            return False
        old_id = target.track_id
        new_boxes: list[PlayerBox] = []
        for b in self.boxes[frame]:
            if b is target:
                new_boxes.append(replace(b, track_id=new_id))
            elif b.track_id == new_id and new_id != UNKNOWN_ID:
                new_boxes.append(replace(b, track_id=old_id))
            else:
                new_boxes.append(b)
        self._store(frame, new_boxes)
        self.mark_reviewed(frame)
        return True

    def swap_ids(self, frame: int, a: int = 1, b: int = 2, *, to_end: bool = False) -> int:
        """Swap ids ``a`` and ``b`` on ``frame`` (or from ``frame`` to the end). Returns frames changed."""
        frames = [f for f in self.boxes if f >= frame] if to_end else [frame]
        changed = 0
        for f in frames:
            boxes = self.boxes.get(f, [])
            if not any(x.track_id in (a, b) for x in boxes):
                continue
            swapped = [
                replace(x, track_id=b)
                if x.track_id == a
                else replace(x, track_id=a)
                if x.track_id == b
                else x
                for x in boxes
            ]
            self._store(f, swapped)
            changed += 1
        if changed:
            self.mark_reviewed(frame)
        return changed

    def copy_previous(self, frame: int) -> bool:
        """Copy the boxes of the nearest previous frame that has boxes (as non-keyframes)."""
        previous = [f for f in self.boxes if f < frame]
        if not previous:
            return False
        source = self.boxes[max(previous)]
        self._store(frame, [replace(b, keyframe=False) for b in source])
        return True

    def mark_keyframes(self, frame: int) -> int:
        """Validate every box of ``frame`` as a keyframe (e.g. a correct pre-annotation)."""
        boxes = self.boxes.get(frame, [])
        self._store(frame, [replace(b, keyframe=True) for b in boxes])
        self.mark_reviewed(frame)
        return len(boxes)

    def interpolate_to(self, frame: int) -> int:
        """For each known id on ``frame``: make it a keyframe and linearly fill the frames
        since the previous keyframe of that id. Returns the number of boxes written."""
        written = 0
        for box in self.frame_boxes(frame):
            if box.track_id == UNKNOWN_ID:
                continue
            end = replace(box, keyframe=True)
            self.set_box(frame, end)
            previous = self.previous_keyframe(frame, box.track_id)
            if previous is None:
                continue
            start_frame, start = previous
            span = frame - start_frame
            for f in range(start_frame + 1, frame):
                self.set_box(f, _lerp_box(start, end, (f - start_frame) / span))
                written += 1
        return written

    # ------------------------------------------------------------------ CSV
    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for frame in sorted(set(self.boxes) | self.reviewed):
            reviewed = int(frame in self.reviewed)
            boxes = sorted(self.boxes.get(frame, []), key=lambda b: b.track_id)
            if not boxes:
                rows.append({"frame": frame, "reviewed": reviewed})
                continue
            for b in boxes:
                rows.append(
                    {
                        "frame": frame,
                        "class_id": b.class_id,
                        "track_id": b.track_id,
                        "x0": round(b.x0, 2),
                        "y0": round(b.y0, 2),
                        "x1": round(b.x1, 2),
                        "y1": round(b.y1, 2),
                        "keyframe": int(b.keyframe),
                        "reviewed": reviewed,
                    }
                )
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        for col in ("class_id", "track_id", "keyframe"):
            df[col] = df[col].astype("Int64")
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PlayerAnnotations:
        ann = cls()
        legacy = "track_id" not in df.columns
        has_reviewed = "reviewed" in df.columns
        for row in df.to_dict("records"):
            frame = int(row["frame"])
            if has_reviewed and _is_set(row.get("reviewed")) and int(row["reviewed"]) == 1:
                ann.reviewed.add(frame)
            coords = [row.get(k) for k in ("x0", "y0", "x1", "y1")]
            if any(not _is_set(v) for v in coords):
                continue
            track_id = UNKNOWN_ID if legacy else _int_or(row.get("track_id"), UNKNOWN_ID)
            box = PlayerBox.from_corners(
                track_id,
                *(float(v) for v in coords),
                class_id=_int_or(row.get("class_id"), 0),
                keyframe=bool(_int_or(row.get("keyframe"), 0)),
            )
            ann.set_box(frame, box, review=not has_reviewed)
        return ann

    def save_csv(self, path: str | Path) -> Path:
        path = Path(path)
        self.to_dataframe().to_csv(path, index=False)
        return path

    @classmethod
    def load_csv(cls, path: str | Path) -> PlayerAnnotations:
        return cls.from_dataframe(pd.read_csv(path))

    # ------------------------------------------------------------------ MOT
    def to_mot_lines(self) -> tuple[list[str], int]:
        """MOT ground-truth lines for reviewed frames; returns (lines, skipped unknown-id boxes)."""
        lines: list[str] = []
        skipped = 0
        for frame in sorted(self.reviewed):
            for b in sorted(self.boxes.get(frame, []), key=lambda x: x.track_id):
                if b.track_id == UNKNOWN_ID:
                    skipped += 1
                    continue
                # Same rounding as the CSV, so gt.txt and the CSV always agree.
                x0, y0, x1, y1 = (round(v, 2) for v in (b.x0, b.y0, b.x1, b.y1))
                lines.append(
                    f"{frame + 1},{b.track_id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},1,1,1.0"
                )
        return lines, skipped

    def export_mot(self, directory: str | Path) -> tuple[Path, int, int]:
        """Write ``<directory>/gt/gt.txt`` and ``labels.txt``. Returns (gt path, boxes, skipped)."""
        gt_dir = Path(directory) / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        lines, skipped = self.to_mot_lines()
        gt_path = gt_dir / "gt.txt"
        gt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (gt_dir / "labels.txt").write_text("player\n", encoding="utf-8")
        return gt_path, len(lines), skipped


def _is_set(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _int_or(value: Any, default: int) -> int:
    return int(value) if _is_set(value) else default


class TrackIdMapper:
    """Turn tracker ids (arbitrary, change after a cut) into player ids 1..n_players.

    Per frame, the ``n_players`` most confident detections are kept. A tracker id
    already seen keeps its player id. New tracker ids get the free player ids:
    with no history, the lowest box in the image (closest to the camera) gets 1;
    otherwise each free id goes to the box whose feet are closest to where that
    player was last seen. This is only a pre-annotation: a human checks it.
    """

    def __init__(self, n_players: int = 2) -> None:
        self.n_players = n_players
        self.mapping: dict[int, int] = {}
        self.last_bottom: dict[int, float] = {}

    def reset(self) -> None:
        self.mapping.clear()
        self.last_bottom.clear()

    def update(
        self,
        detections: Iterable[tuple[int | None, float, Sequence[float]]],
    ) -> list[PlayerBox]:
        """``detections``: (tracker_id or None, confidence, xyxy)."""
        top = sorted(detections, key=lambda d: d[1], reverse=True)[: self.n_players]
        assigned: dict[int, tuple[float, ...]] = {}
        pending: list[tuple[int | None, tuple[float, ...]]] = []
        for tracker_id, _conf, xyxy in top:
            xyxy_t = tuple(float(v) for v in xyxy)
            pid = self.mapping.get(tracker_id) if tracker_id is not None else None
            if pid is not None and pid not in assigned:
                assigned[pid] = xyxy_t
            else:
                pending.append((tracker_id, xyxy_t))

        free = [p for p in range(1, self.n_players + 1) if p not in assigned]
        if pending and free:
            if all(p not in self.last_bottom for p in free):
                pending.sort(key=lambda d: d[1][3], reverse=True)  # lowest in image first
                pairs = list(zip(free, pending, strict=False))
            else:
                pairs = []
                remaining = list(pending)
                for pid in free:
                    if not remaining:
                        break
                    ref = self.last_bottom.get(pid)
                    if ref is None:
                        best = max(remaining, key=lambda d: d[1][3])
                    else:
                        best = min(remaining, key=lambda d: abs(d[1][3] - ref))
                    remaining.remove(best)
                    pairs.append((pid, best))
            for pid, (tracker_id, xyxy_t) in pairs:
                assigned[pid] = xyxy_t
                if tracker_id is not None:
                    self.mapping[tracker_id] = pid

        boxes = []
        for pid, xyxy_t in sorted(assigned.items()):
            self.last_bottom[pid] = xyxy_t[3]
            boxes.append(PlayerBox.from_corners(pid, *xyxy_t))
        return boxes
