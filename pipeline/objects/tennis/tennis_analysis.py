

from pipeline.objects.analysis import Analysis
from pipeline.objects.tennis.tennis_frame import TennisFrame


class TennisAnalysis(Analysis[TennisFrame]):
    frame_class = TennisFrame

