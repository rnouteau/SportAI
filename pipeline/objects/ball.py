class Ball:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, position: tuple[float, float] | list[float], confidence: float) -> "Ball":
        return cls({"position": list(position), "confidence": confidence})

    @property
    def position(self) -> list[float] | None:
        return self.data.get("position", None)

    @property
    def confidence(self) -> float | None:
        return self.data.get("confidence", None)
