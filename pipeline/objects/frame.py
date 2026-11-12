class Frame:
    def __init__(self, data: dict) -> None:
        self.data = data

    @property
    def id(self) -> int | None:
        return self.data.get("id", None)

    @id.setter
    def id(self, value: int) -> None:
        self.data["id"] = value
