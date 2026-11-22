from pipeline.objects.player_at_frame import PlayerAtFrame

NEAR, FAR = "near", "far"
SIDES = (NEAR, FAR)


class TennisPlayerAtFrame(PlayerAtFrame):
    def __init__(self, data: dict) -> None:
        super().__init__(data)

    @property
    def side(self) -> str | None:
        """Half of the court on this frame: ``"near"`` (court y < 0) or ``"far"`` (y > 0)."""
        side = self.data.get("side")
        return str(side) if side is not None else None

    @side.setter
    def side(self, side: str | None) -> None:
        if side is None:
            self.data.pop("side", None)
            return
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.data["side"] = side
