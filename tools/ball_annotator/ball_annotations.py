"""Ball annotations: positions and events per frame, CSV I/O (no GUI).

Used by ``tools/ball_annotator/ball_anotator.py``. Kept free of Tk / OpenCV so it can be
unit-tested.

CSV (``<video>.csv``, next to the video, read by the ball evaluator):
    frame, x, y, event
- one row per frame of the video: every frame counts as reviewed;
- ``x`` / ``y`` in original video pixels, empty when the ball is not visible;
- ``event`` empty, or one of ``EVENTS``. An event can be set on a frame without a ball
  position (ball hidden by the player at the hit, for instance);
- legacy CSVs (``frame, x, y`` only) load with no event.

Event conventions (frame of the contact):
- ``serve``  : the server hits the ball (first shot of a point);
- ``hit``    : any other racket contact (groundstroke, volley, smash, return...);
- ``bounce`` : the ball touches the ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EVENTS = ("serve", "hit", "bounce")
CSV_COLUMNS = ["frame", "x", "y", "event"]


@dataclass
class BallAnnotations:
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    events: dict[int, str] = field(default_factory=dict)

    # ---- edits -------------------------------------------------------------------------

    def set_position(self, frame: int, x: float, y: float) -> None:
        self.positions[frame] = (float(x), float(y))

    def delete_position(self, frame: int) -> bool:
        """Remove the ball of ``frame``; ``True`` if there was one."""
        return self.positions.pop(frame, None) is not None

    def toggle_event(self, frame: int, event: str) -> str | None:
        """Set ``event`` on ``frame``, or remove it if it is already there.

        Returns the event of the frame after the call (``None`` when removed)."""
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}, expected one of {EVENTS}")
        if self.events.get(frame) == event:
            del self.events[frame]
            return None
        self.events[frame] = event
        return event

    def clear_event(self, frame: int) -> bool:
        """Remove the event of ``frame``; ``True`` if there was one."""
        return self.events.pop(frame, None) is not None

    # ---- queries -----------------------------------------------------------------------

    def next_event(self, frame: int) -> int | None:
        """First frame after ``frame`` holding an event."""
        return min((f for f in self.events if f > frame), default=None)

    def previous_event(self, frame: int) -> int | None:
        """Last frame before ``frame`` holding an event."""
        return max((f for f in self.events if f < frame), default=None)

    def event_counts(self) -> dict[str, int]:
        """Number of annotated events of each kind, in ``EVENTS`` order."""
        values = list(self.events.values())
        return {event: values.count(event) for event in EVENTS}

    # ---- CSV ---------------------------------------------------------------------------

    def to_dataframe(self, frame_count: int) -> pd.DataFrame:
        """One row per frame (``0 .. frame_count - 1``, more if something is annotated after)."""
        last = max([frame_count - 1, *self.positions, *self.events], default=-1)
        rows = []
        for frame in range(last + 1):
            x, y = self.positions.get(frame, (None, None))
            rows.append({"frame": frame, "x": x, "y": y, "event": self.events.get(frame)})
        return pd.DataFrame(rows, columns=CSV_COLUMNS).astype({"x": float, "y": float})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> BallAnnotations:
        annotations = cls()
        has_event = "event" in df.columns
        for row in df.itertuples(index=False):
            frame = int(row.frame)
            if not (pd.isna(row.x) or pd.isna(row.y)):
                annotations.set_position(frame, row.x, row.y)
            if has_event and not pd.isna(row.event) and str(row.event).strip():
                event = str(row.event).strip().lower()
                if event not in EVENTS:
                    raise ValueError(f"frame {frame}: unknown event {row.event!r}")
                annotations.events[frame] = event
        return annotations

    def save_csv(self, path: Path, frame_count: int) -> None:
        self.to_dataframe(frame_count).to_csv(path, index=False)

    @classmethod
    def load_csv(cls, path: Path) -> BallAnnotations:
        # keep_default_na=False + na_values=[""]: an event is never read as NaN by accident
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        return cls.from_dataframe(df)
