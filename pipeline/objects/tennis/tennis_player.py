from pipeline.objects.player import Player


class TennisPlayer(Player):
    """Tennis player over the whole video, with the tracker ids that followed him."""

    @property
    def track_ids(self) -> list[int]:
        return [int(v) for v in self.data.get("track_ids") or []]

    @track_ids.setter
    def track_ids(self, value: list[int]) -> None:
        self.data["track_ids"] = [int(v) for v in value]
