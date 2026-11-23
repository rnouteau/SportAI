from dataclasses import dataclass, field

from pipeline.objects.analysis import Analysis
from pipeline.objects.tennis.tennis_court_schema import TennisCourtSchema
from pipeline.objects.tennis.tennis_final_stats import TennisFinalStats
from pipeline.objects.tennis.tennis_frame import TennisFrame


@dataclass
class TennisAnalysis(Analysis[TennisFrame]):
    frame_class = TennisFrame
    final_stats_class = TennisFinalStats
    final_stats: TennisFinalStats = field(default_factory=TennisFinalStats)

    def __post_init__(self) -> None:
        self.court_schema = TennisCourtSchema()
