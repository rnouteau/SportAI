import pytest

from tests.assets import resolve_dataset_dir
from tests.evaluation.court_detection_evaluation import CourtDetectionEvaluator
from tests.nrt.harness import assert_vs_baseline, base_context, prepare_config
from utils.paths import get_root

config_file = (
    get_root()
    / "tests"
    / "configs"
    / "nrt"
    / "tennis"
    / "config_test_tennis_accuracy_court_detection.yaml"
)
report_dir = get_root() / "outputs" / "tests" / "tennis_court_detection"

COUNT_COLUMNS = ["Video", "Total", "TP", "FP", "FN"]
REPORT_TITLE = "Tennis court detection - non-regression"


def run_context(config: dict) -> dict:
    evaluation = config["evaluation"]
    return {
        **base_context(config),
        "match_threshold": evaluation.get("match_threshold", 0.05),
        "oks_sigma": evaluation.get("oks_sigma", 0.05),
    }


@pytest.mark.nrt
def test_tennis_court_detection_regression(update_baseline: bool) -> None:
    config = prepare_config(config_file)

    results = CourtDetectionEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
    assert len(results) >= 2, "no annotated court image found"

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
    CourtDetectionEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
