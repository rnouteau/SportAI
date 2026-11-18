from pipeline.objects.frame import Frame
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame


class TennisFrame(Frame):
    def __init__(self, data: dict) -> None:
        super().__init__(data)

    @property
    def players(self) -> list[TennisPlayerAtFrame]:
        player_data_list = self.data.get("players", [])
        return [TennisPlayerAtFrame(data) for data in player_data_list]

    @players.setter
    def players(self, players: list[TennisPlayerAtFrame] | None) -> None:
        self.data["players"] = [player.data if player else None for player in (players or [])]

    @property
    def ball(self) -> TennisBall | None:
        ball_data = self.data.get("ball", None)
        return TennisBall(ball_data) if ball_data else None

    @ball.setter
    def ball(self, ball: TennisBall | None) -> None:
        self.data["ball"] = ball.data if ball else None

    @property
    def homography(self) -> TennisHomography | None:
        homography_data = self.data.get("homography")
        return TennisHomography(homography_data) if homography_data else None

    @homography.setter
    def homography(self, homography: TennisHomography | None) -> None:
        self.data["homography"] = homography.data if homography else None
