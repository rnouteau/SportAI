from dataclasses import dataclass, field
from typing import Self

from pipeline.objects.frame import Frame
from pipeline.objects.tennis.tennis_ball import TennisBall
from pipeline.objects.tennis.tennis_homography import TennisHomography
from pipeline.objects.tennis.tennis_player_at_frame import TennisPlayerAtFrame


@dataclass
class TennisFrame(Frame):
    players: list[TennisPlayerAtFrame] = field(default_factory=list)
    ball: TennisBall | None = None
    homography: TennisHomography | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        players = [TennisPlayerAtFrame.from_dict(p) for p in (data.get("players") or [])]
        ball = TennisBall.from_dict(data["ball"]) if data.get("ball") else None
        homography = (
            TennisHomography.from_dict(data["homography"]) if data.get("homography") else None
        )
        return cls(id=data["id"], players=players, ball=ball, homography=homography)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["players"] = [p.to_dict() for p in self.players]
        d["ball"] = self.ball.to_dict() if self.ball else None
        d["homography"] = self.homography.to_dict() if self.homography else None
        return d
