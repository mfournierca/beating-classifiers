import numpy as np
import pandas as pd
from random import uniform
from scipy.optimize import minimize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class AntiClassifier(object):

    def __init__(self, classifier, feature_specs):
        """A machine which attacks a given classifier, looking for 
        misclassification errors.
        
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
        self._transform = FeatureUnion([("scaler", StandardScaler())])
        self.anticlassifier = Pipeline(
            steps=[
                ("transform", self._transform), 
                ("logistic", LogisticRegression())
            ]
        ) 
        self.prepare()

    def prepare(self, num_points=10000):
        """Prepare the anti model. 

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
        df = pd.DataFrame(rows)

        # feed to the model to build a training set
        p = self.classifier.predict(df)

        # fit the antimodel       
        self.anticlassifier.fit(df, p)  

    def coefs(self):
        """Get the coefficients of the anticlassifier's decision function"""
        return self.anticlassifier.get_params()["logistic"].coef_[0]

    def decision_function(self, x):
        """Return the decision function of the anticlassifier evaluated at x"""
        assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
        return self.anticlassifier.decision_function(x)

    def decision_gradient(self, x):
        """Return the gradient of the anticlassifier decision function 
        evaluated at x. x must be the raw feature vector, ie not normalized or 
        transformed.
        """
        assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
        coefs = self.coefs()
        t = self._transform.transform(x)
        g = [
            # the sigmoid partial derivative
            (c * np.exp(t.dot(coefs))) / ((1.0 + np.exp(t.dot(coefs)))**2.0)
            for c in coefs
        ]
        return np.array(g)

    def minimize_decision_function(self, constraints):
        """Minimize the decision function of the antimodel under constraints. 
        Return the feature vector which minimizes the function.""" 

        # the constraints are expressed as boundaries, not values
        # we're minimizing within a region
 
        x = minimize(
            self.decision_function, 
            self.guess(constraints),
            method="SLSQP",
            jac=self.decision_gradient,
            bounds=[(x["min"], x["max"]) for x in self.feature_specs],
            constraints=constraints,
            tol=0.001
        )
        return x

    def guess(self, constraints):
        """Take an initial guess at the a feature vector under constraints"""
        r = [
            f["type"](uniform(f["min"], f["max"])) for f in self.feature_specs 
        ]
        for c in constraints:
            c["init"](r)
        return np.array(r)

    def run():
        pass
