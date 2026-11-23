from dataclasses import dataclass

from pipeline.objects.final_stats import FinalStats
from pipeline.objects.tennis.tennis_player import TennisPlayer


@dataclass
class TennisFinalStats(FinalStats[TennisPlayer]):
    player_class = TennisPlayer
