
from pathlib import Path

from tests.base_test import BaseTest
from pipeline.run import run_pipeline
from utils import read_yaml
from loguru import logger
import pandas as pd
import os

ROOT = Path(__file__).resolve().parents[3] 
config_file = ROOT / "tests" / "configs" / "nrt" / "tennis" / "config_test_tennis_accuracy_ball_tracking.yaml"
directory = ROOT / "datasets" / "tennis" / "ball_detection" / "test"


class TestTennisAccuracyBallTracking:
    def __init__(self,config_file,directory):
        logger.info("Initialized TestTennisAccuracyBallTracking")
        self.config = read_yaml(config_file)
        self.directory = directory
        self.reset()

    def run_test(self):
        for video_path in Path(self.directory).rglob("*.mp4"):
            result = run_pipeline("tennis", str(video_path), self.config, test=True)
            self.evaluate(result, os.path.join(directory,video_path.with_suffix(".csv")))
        
        self.print_result(self)

    def reset(self):
        self.tp, self.fp1, self.fp2, self.fp, self.tn, self.fn = 0, 0, 0, 0, 0, 0

    def print_result(self):
        pass

    def evaluate(self,result,ground_truth):
        frames : dict = result["frames"]
        df = pd.read_csv(ground_truth)

        print(frames)
        print(df.head(5))
        eps = 1e-15
        self.fp = self.fp1 + self.fp2
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {
            'Total': total,
            'TP': self.tp,
            'FP1': self.fp1,
            'FP2': self.fp2,
            'FP': self.fp,
            'TN': self.tn,
            'FN': self.fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }


if __name__ == "__main__":
    test = TestTennisAccuracyBallTracking(config_file,directory)
    test.run_test()
