import pytest

from tests.assets import resolve_dataset_dir
from tests.evaluation.ball_detection_evaluation import BallTrackingEvaluator
from tests.nrt.harness import assert_vs_baseline, base_context, prepare_config
from utils.paths import get_root

config_file = (
    get_root()
    / "tests"
    / "configs"
    / "nrt"
    / "tennis"
    / "config_test_tennis_accuracy_ball_detection.yaml"
)
report_dir = get_root() / "outputs" / "tests" / "tennis_ball_detection"

COUNT_COLUMNS = ["Video", "Total", "TP", "FP", "TN", "FN"]
REPORT_TITLE = "Tennis ball tracking - non-regression"


def run_context(config: dict) -> dict:
    model_cfg = config["models"][0]["config"]
    return {
        **base_context(config),
        "threshold": model_cfg.get("threshold", 0.5),
        "min_dist": config["evaluation"]["min_dist"],
    }


@pytest.mark.nrt
def test_tennis_ball_tracking_regression(update_baseline: bool) -> None:
    config = prepare_config(config_file)

    results = BallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
    assert len(results) > 1, "no annotated video found"

    assert_vs_baseline(
        results,
        config,
        run_context(config),
        report_dir,
        title=REPORT_TITLE,
        count_columns=COUNT_COLUMNS,
        update_baseline=update_baseline,
    )


if __name__ == "__main__":
    config = prepare_config(config_file)
    BallTrackingEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
