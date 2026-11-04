

from pipeline.Analysis import Analysis


class TennisAnalysis(Analysis):
    def __init__(self, config):
        super().__init__(config)
        self.create_frame()

