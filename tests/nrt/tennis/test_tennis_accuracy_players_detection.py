import pytest

from tests.assets import resolve_dataset_dir
from tests.evaluation.player_detection_evaluation import PlayerDetectionEvaluator
from tests.nrt.harness import assert_vs_baseline, base_context, prepare_config
from utils.paths import get_root

config_file = (
    get_root()
    / "tests"
    / "configs"
    / "nrt"
    / "tennis"
    / "config_test_tennis_accuracy_players_detection.yaml"
)
report_dir = get_root() / "outputs" / "tests" / "tennis_player_detection"

COUNT_COLUMNS = ["Video", "Total", "TP", "FP", "TN", "FN"]
REPORT_TITLE = "Tennis player detection - non-regression"


def run_context(config: dict) -> dict:
    return {
        **base_context(config),
        "iou_threshold": config["evaluation"]["iou_threshold"],
    }


@pytest.mark.nrt
def test_tennis_player_detection_regression(update_baseline: bool) -> None:
    config = prepare_config(config_file)

    results = PlayerDetectionEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
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
    PlayerDetectionEvaluator(config, resolve_dataset_dir(config["dataset"])).run_test()
