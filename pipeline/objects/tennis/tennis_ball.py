

from pipeline.objects.ball import Ball


class TennisBall(Ball):
    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def velocity(self) -> float | None:
        return self.data.get("velocity", None)

    @velocity.setter
    def velocity(self, value : float | None):
        self.data["velocity"] = value
