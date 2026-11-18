class Frame:
    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def empty(cls, frame_id: int) -> dict:
        """Sport-agnostic empty frame dict (override in sport Frame subclasses)."""
        return {"id": frame_id}

    @property
    def id(self) -> int:
        return self.data["id"]

    @id.setter
    def id(self, value: int) -> None:
        self.data["id"] = value
