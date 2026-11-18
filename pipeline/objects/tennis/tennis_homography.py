from pipeline.objects.homography import Homography


class TennisHomography(Homography):
    def __init__(self, data: dict) -> None:
        super().__init__(data)

    @property
    def keypoints(self) -> list[list[float]] | None:
        return self.data.get("keypoints")

    @property
    def confidences(self) -> list[float] | None:
        return self.data.get("confidences")
