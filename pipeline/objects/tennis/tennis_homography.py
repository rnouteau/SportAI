from pipeline.objects.homography import Homography


class TennisHomography(Homography):
    def __init__(self, data: dict) -> None:
        super().__init__(data)
