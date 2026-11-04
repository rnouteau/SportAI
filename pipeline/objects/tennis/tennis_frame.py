

from pipeline.objects.frame import Frame
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame


class TennisFrame(Frame):
    def __init__(self, data : dict):
        super().__init__(data)
    
    @property
    def players(self) -> list[TennisPlayerAtFrame]:
        player_data_list = self.data.get("players", [])
        return [TennisPlayerAtFrame(data) for data in player_data_list]

    @property
    def ball(self) -> TennisBall | None:
        ball_data = self.data.get("ball", None)
        return TennisBall(ball_data) if ball_data else None

    @ball.setter
    def ball(self, ball: TennisBall | None):
        self.data["ball"] = ball.data if ball else None

