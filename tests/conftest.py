import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--update-baseline", action="store_true", help="Rewrite the NRT baselines")


@pytest.fixture
def update_baseline(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--update-baseline")
