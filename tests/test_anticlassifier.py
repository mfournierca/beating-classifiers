from unittest import TestCase
from mock import patch, MagicMock
import numpy as np

from src import anticlassifier

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

    def predict(self, df):
        # only one dim matters for the test classifier
        r = [int(x[1][0] > 0) for x in df.iterrows()]
        return r


class TestAntiClassifier(TestCase):

    def runTest(self):
        pass

    def setUp(self):
        self.feature_specs = TEST_FEATURE_SPECS
        self.a = anticlassifier.AntiClassifier(
            _TestClassifier(), self.feature_specs
        )

    def test_prepare_anticlassifier(self):
        self.a.prepare(num_points=1000)
        # should not raise an error
        self.a.anticlassifier.predict(
            np.array([0 for i in range(len(self.feature_specs))])
        )

    def test_coefs_created(self):
        self.assertTrue(self.a.coefs)

    def test_decision_function_value(self):
        self.assertGreater(
            self.a.decision_function(
                np.array([0 for i in range(len(self.feature_specs))])
            ),
            0.0
        )

    def test_decision_gradient_value(self):
        self.assertGreater(
            self.a.decision_gradient(
                np.array([0 for i in range(len(self.feature_specs))])
            ),
            0.0
        )

