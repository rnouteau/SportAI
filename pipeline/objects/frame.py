from dataclasses import dataclass
from typing import Self


@dataclass
class Frame:
    id: int

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(id=data["id"])

    def to_dict(self) -> dict:
        return {"id": self.id}
