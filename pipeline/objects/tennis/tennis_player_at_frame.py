
from pipeline.objects.player_at_frame import PlayerAtFrame


class TennisPlayerAtFrame(PlayerAtFrame):
    def __init__(self, data: dict):
        super().__init__(data)
        