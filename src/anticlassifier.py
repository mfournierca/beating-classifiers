import numpy as np
import pandas as pd
from random import uniform

import scipy.optimize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class AntiClassifier(object):

    xtrain = None
    ytrain = None
    new_rows = None
    retrain_interval = 100

    def __init__(
            self,
            classifier,
            feature_specs,
            prepare=True,
            num_points=10000
        ):
        """A machine which attacks a given classifier, looking for
        misclassification errors.

        Internally this class uses an sklearn pipeline to transform data and
        a logistical regression classifier to model the behaviour of the
        classifiern under attack.

        :param classifier: an sklearn classifier which is already trained.
            The predict() method must work.
        :type m: object
        :param feature_specs: Specifications for the features that the
            anti-classifier will use. Must be a list of dictionaries of the
            form:
            {
                "name": string
                "type": float or int
                "min": min value
                "max": max value
            }
        :type feature_specs: list
        """
        self.classifier = classifier
        self.feature_specs = feature_specs
        self._transform = Pipeline(steps=[("scaler", StandardScaler())])
        self.anticlassifier = Pipeline(
            steps=[
                ("transform", self._transform),
                ("logistic", LogisticRegression(fit_intercept=True))
            ]
        )
        if prepare:
            self.prepare(num_points=num_points)

    def prepare(self, num_points=10000):
        """Prepare the anticlassifier.

        Randomly generate a new test set of feature vectors and pass it to the
        classifier. Use the output as the labels to train the anticlassifier.

        :param num_points: the number of points used to train the anticlassfier
        :type num_points: int
        """

        # generate an initial set of random feature vectors
        rows = [
            {
                c["name"]: c["type"](uniform(c["min"], c["max"]))
                for c in self.feature_specs
            }
            for i in range(num_points)
        ]

        # self.xtrain is stored and will grow with the number of predictions
        # the anticlassifier makes, see self.get() and self.retrain()
        self.xtrain = pd.DataFrame(rows)

        # feed to the classifier to build a training set
        self.ytrain = self.classifier.predict(self.xtrain)

        # fit the anticlassifier
        self.anticlassifier.fit(self.xtrain, self.ytrain)

        # used to periodically update / retrain the anticlassifier
        self.new_rows = pd.DataFrame(columns=self.xtrain.columns)

    def _retrain(self):
        """Retrain the anticlassifier using the guesses that it generated"""
        y = self.classifier.predict(self.new_rows)
        self.xtrain = pd.concat([self.xtrain, self.new_rows])
        self.ytrain = np.append(self.ytrain, y)
        self.anticlassifier.fit(self.xtrain, self.ytrain)
        self.new_rows = pd.DataFrame(columns=self.xtrain.columns)

    def _lg_coefs(self):
        """Get the coefficients of the logisitical regression classifier used
        by the anticlassifier."""
        return self.anticlassifier.get_params()["logistic"].coef_[0]

    def _lg_intercept(self):
        """Get the intercept term of the logisitical regression classifier used
        by the anticlassifier."""
        return self.anticlassifier.get_params()["logistic"].intercept_

    def _lg_classes(self):
        """Get the class labels of the logisitical regression classifier
        used by the anticlassifier."""
        return self.anticlassifier.get_params()["logistic"].classes_

    def _lg_predict_proba(self, x):
        """Return the predict_proba method evaluated at x of the logistical
        regression classifier used by the anticlassifier.

        x is not normalized or transformed before computing the function.
        """
        assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
        l = self.anticlassifier.get_params()["logistic"]
        return l.predict_proba(x)[0][1]

    def _lg_predict_proba_gradient(self, x):
        """Return the gradient evaluated at x of the logistic regression
        classifier used by the anticlassifier.

        x is not normalized or transformed before computing the gradient.
        """
        assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
        coefs = self._lg_coefs()
        inter = self._lg_intercept()
        g = [
            # the sigmoid partial derivative
            (
                (c * np.exp(inter + x.dot(coefs))) /
                ((1.0 + np.exp(inter + x.dot(coefs)))**2.0)
            )
            for c in coefs
        ]
        return np.array(g)

    def _minimize(self, constraints):
        """Find the feature vector which minimizes the decision function of the
        anticlassifier under constraints. Return this feature vector.

        The anticlassifier performs a normalization on the training data. This
        function runs the inverse transform on the feature vector before
        returning it, i.e. we return the "raw", unnormalized feature vector.
        """

        # the constraints are expressed as boundaries, not values
        # we're minimizing within a region

        result = scipy.optimize.minimize(
            self._lg_predict_proba,
            self._transform.transform(self._guess(constraints)),
            method="SLSQP",
            jac=self._lg_predict_proba_gradient,
            bounds=[(x["min"], x["max"]) for x in self.feature_specs],
            constraints=constraints,
            tol=0.001
        )
        return self._transform.inverse_transform(result.x)[0]

    def _guess(self, constraints):
        """Take an initial guess at the a feature vector under constraints"""
        r = [
            f["type"](uniform(f["min"], f["max"])) for f in self.feature_specs
        ]
        for c in constraints:
            c["init"](r)
        return np.array(r)

    def get(self, constraints):
        """Get a feature vector which satisfies the provided constraints and
        which the classifier is expected to accept.

        Also retrain the anticlassifier if enough feature vectors have been
        generated."""
        v = self._minimize(constraints)
        self.new_rows.loc[len(self.new_rows) + 1] = v
        if len(self.new_rows) >= self.retrain_interval:
            self._retrain()
        return v

