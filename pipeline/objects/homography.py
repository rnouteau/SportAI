from dataclasses import dataclass
from typing import Self


@dataclass
class Homography:
    keypoints: list[list[float]] | None = None
    confidences: list[float] | None = None
    matrix: list[list[float]] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            keypoints=data.get("keypoints"),
            confidences=data.get("confidences"),
            matrix=data.get("matrix"),
        )

    def to_dict(self) -> dict:
        """Fields left at None are not written (e.g. no matrix when RANSAC failed)."""
        d: dict = {}
        if self.keypoints is not None:
            d["keypoints"] = [[float(kp[0]), float(kp[1])] for kp in self.keypoints]
        if self.confidences is not None:
            d["confidences"] = [float(c) for c in self.confidences]
        if self.matrix is not None:
            d["matrix"] = [[float(v) for v in row] for row in self.matrix]
        return d
