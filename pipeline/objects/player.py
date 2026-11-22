from typing import Self


class Player:
    """A player over the whole video (JSON-backed), e.g. in ``final_stats["players"]``."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def create(cls, **fields: object) -> Self:
        return cls(dict(fields))

    @property
    def id(self) -> int | None:
        value = self.data.get("id")
        return int(value) if value is not None else None

    @id.setter
    def id(self, value: int | None) -> None:
        if value is None:
            self.data.pop("id", None)
        else:
            self.data["id"] = int(value)

    @property
    def name(self) -> str | None:
        return self.data.get("name")

    @name.setter
    def name(self, value: str | None) -> None:
        if value is None:
            self.data.pop("name", None)
        else:
            self.data["name"] = str(value)
