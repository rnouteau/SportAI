

from pipeline.objects.ball import Ball


class TennisBall(Ball):
    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def velocity(self):
        return self.data.get("velocity", None)

    @property
    def position(self):
        return self.data.get("position",None)

    @position.setter
    def position(self, value):
        self.data["position"] = value