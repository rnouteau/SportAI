from pipeline.objects.analysis import Analysis
from pipeline.objects.tennis.tennis_court_schema import TennisCourtSchema
from pipeline.objects.tennis.tennis_frame import TennisFrame


class TennisAnalysis(Analysis[TennisFrame]):
    frame_class = TennisFrame

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self.court_schema = TennisCourtSchema()
