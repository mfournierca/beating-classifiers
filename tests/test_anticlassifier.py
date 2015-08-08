from unittest import TestCase
from mock import patch, MagicMock
import numpy as np
import copy

from src import anticlassifier

TEST_FEATURE_SPECS = [
    {
        "name": "test_var_1",
        "type": float,
        "max": 1.0,
        "min": -1.0
    },
    {
        "name": "test_var_2",
        "type": float,
        "max": 1.0,
        "min": -1.0
    }
]

TEST_VAR_1_INIT = 0.25
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

    def test_prepare(self):
        # if self.a.prepare() was run, this should not raise an error
        self.a.anticlassifier.predict(
            np.array([0 for i in range(len(self.feature_specs))])
        )

    def test_lg_classes(self):
        self.assertEqual(list(self.a._lg_classes()), [0, 1])

    def test_lg_coefs(self):
        self.assertTrue(self.a._lg_coefs().all())

    def test_lg_intercept(self):
        i = self.a._lg_intercept()
        self.assertTrue(i)
        self.assertGreater(i, -1)
        self.assertLess(i, 1)

    def test_lg_predict_proba(self):
        features = np.array([0.0 for i in range(len(self.feature_specs))])
        v = self.a._lg_predict_proba(features)
        c = self.a._lg_coefs()
        i = self.a._lg_intercept()
        e = 1.0 / (1.0 + np.exp(-1.0 * (i + c.dot(features))))

        self.assertGreater(v, 0.0, "value must be greater than 0")
        self.assertLess(v, 1.0, "value must be less than 1")
        self.assertEqual(v, e, "value != sigmoid output: {0} != {1}".format(
            v, e))

    def test_lg_predict_proba_gradient(self):
        f = np.array([0.0 for i in range(len(self.feature_specs))])

        # sanity check gradient
        v = self.a._lg_predict_proba_gradient(f)
        self.assertGreater(v.all(), 0.0)

        # check approximate gradient
        h = 0.0001
        m = np.array([f[0] + h, f[1]])
        n = np.array([f[0], f[1] + h])
        a = np.array([
            [(self.a._lg_predict_proba(m) - self.a._lg_predict_proba(f)) / h],
            [(self.a._lg_predict_proba(n) - self.a._lg_predict_proba(f)) / h]
        ])

        r = v - a
        self.assertTrue(
            (np.abs(r) < np.array([0.001, 0.001])).all(),
            "approximate gradient not accurate enough, diff: {0}".format(r)
        )

    def test_minimize(self):
        pass

