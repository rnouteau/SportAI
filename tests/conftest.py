import pytest


def pytest_addoption(parser):
    parser.addoption("--update-baseline", action="store_true", help="Réécrit les baselines NRT")


@pytest.fixture
def update_baseline(request):
    return request.config.getoption("--update-baseline")
