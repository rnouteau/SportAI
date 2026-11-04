class Ball:
    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def create(cls,position,confidence):
        return cls({"position": list(position), "confidence": confidence})

    @property
    def position(self) -> list | None:
        return self.data.get("position", None)

    @property
    def confidence(self) -> float | None:
        return self.data.get("confidence", None)