from pipeline.objects.final_stats import FinalStats
from pipeline.objects.tennis.tennis_player import TennisPlayer


class TennisFinalStats(FinalStats[TennisPlayer]):
    player_class = TennisPlayer
