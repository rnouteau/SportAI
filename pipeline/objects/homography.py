from typing import Self


class Homography:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(
        cls,
        keypoints: list[list[float]] | list[tuple[float, float]],
        confidences: list[float],
    ) -> Self:
        pts: list[list[float]] = []
        for keypoint in keypoints:
            if len(keypoint) < 2:
                raise ValueError(f"Each keypoint must be [x, y], got {keypoint!r}")
            pts.append([float(keypoint[0]), float(keypoint[1])])
        if len(confidences) != len(pts):
            raise ValueError(f"Expected {len(pts)} confidences, got {len(confidences)}")
        return cls(
            {
                "keypoints": pts,
                "confidences": [float(c) for c in confidences],
            }
        )
