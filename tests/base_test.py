from abc import ABC


class BaseTest(ABC):
    def setUp(self):
        super().setUp()
        # Common setup code for all tests can go here
        self.common_variable = "This is a common variable for all tests"