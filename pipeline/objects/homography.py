from typing import Self


class Homography:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        """Build from free kwargs; new fields (e.g. matrix) use setters."""
        return cls(dict(fields))

    @property
    def keypoints(self) -> list[list[float]] | None:
        return self.data.get("keypoints")

    @keypoints.setter
    def keypoints(self, value: list[list[float]] | list[tuple[float, float]] | None) -> None:
        if value is None:
            self.data.pop("keypoints", None)
            return
        pts: list[list[float]] = []
        for keypoint in value:
            if len(keypoint) < 2:
                raise ValueError(f"Each keypoint must be [x, y], got {keypoint!r}")
            pts.append([float(keypoint[0]), float(keypoint[1])])
        self.data["keypoints"] = pts

    @property
    def confidences(self) -> list[float] | None:
        return self.data.get("confidences")

    @confidences.setter
    def confidences(self, value: list[float] | None) -> None:
        if value is None:
            self.data.pop("confidences", None)
        else:
            self.data["confidences"] = [float(c) for c in value]

    @property
    def matrix(self) -> list[list[float]] | None:
        return self.data.get("matrix")

    @matrix.setter
    def matrix(self, value: list[list[float]] | None) -> None:
        if value is None:
            self.data.pop("matrix", None)
        else:
            self.data["matrix"] = [[float(v) for v in row] for row in value]
