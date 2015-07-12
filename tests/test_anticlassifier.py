from unittest import TestCase
from mock import patch, MagicMock
from ..src import anticlassifier

TEST_FEATURE_SPECS = [
    {
        "name": "test_var_1",
        "type": float,
        "max": 1.0,
        "min": -1.0
    }
]

TEST_VAR_1_INIT = 0.5
TEST_CONSTRAINTS = [
    {
        "name": "test_var_1_gt_0",
        "type": "ineq",
        "fun": lambda x: x,
        "init": lambda x: TEST_VAR_1_INIT
    }
]


class _TestClassifier(object):
    
    def predict(self, x):
        return int(x > 0)


class TestAntiClassifier(TestCase):
    pass
    
    def setUp(self):
        self.a = anticlassifier.AntiClassifier(
            _TestClassifier(), TEST_FEATURE_SPECS
        )
 
    def test_prepare(self):
        pass

