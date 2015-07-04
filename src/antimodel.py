import numpy as np
import pandas as pd
from random import uniform, randint

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src import data


# In the final version of this module we won't be tightly coupled to the 
# features of the model that we are attacking. The decoupling would require
# a invertible mapping between feature vectors and objects (e.g. email 
# messages) which would make matching feature column names and order 
# irrelevant. We don't have that mapping yet so we rely on our features 
# having the same order and names as the model we attack.
xtrain, xtest, ytrain, test = data.load_spambase_test_train()
SPAMBASE_FEATURE_SPECS = [
    {
        "name": c,
        "type": int if type(xtrain[c].ix[0]) == np.int64 else float,
        "max": max(xtrain[c]),
        "min": min(xtrain[c])
    }
    for c in xtrain.columns if c != "spam"
]

# the constraints depend on the order of the feature specs
SPAMBASE_CONSTRAINTS = []
assert xtrain.columns[0] == "capital_run_length_average"
SPAMBASE_CONSTRAINTS.append({
    "name": "capital_run_length_average_gt_10",
    "type": "ineq",
    "fun": lambda x: x[0] - 10,
    "init": lambda x: x.__setitem__(0, randint(10, 25))
})


class AntiModel(object):

    def __init__(self, m, feature_specs=SPAMBASE_FEATURE_SPECS):
        """A class which attacks a given model, looking for misclassification
        errors.

        :param m: an sklearn model which is already trained
        :type m: object
        :param feature_specs: Specifications for the features that the
            antimodel will use. Must be a list of dictionaries of the form:
            {
                "name": string
                "type": float or int
                "min": min value
                "max": max value
            }
        :type feature_specs: list
        """
        self.model = m
        self.feature_specs = feature_specs 
        self._transform = FeatureUnion([("scaler", StandardScaler())])
        self.antimodel = Pipeline(
            steps=[
                ("transform", self._transform), 
                ("logistic", LogisticRegression())
            ]
        ) 
        self.prepare()

    def prepare(self, num_points=10000):
        """Prepare the anti model. 

        :param num_points: the number of points used to train the antimodel
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
        p = self.model.predict(df)

        # fit the antimodel       
        self.antimodel.fit(df, p)  

    def coefs(self):
        """Get the coefficients of the antimodel's decision function"""
        return self.antimodel.get_params()["logistic"].coef_[0]

    def decision_function(self, x):
        """Return the decision function of the anti model evaluated at x."""
        assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
        return self.antimodel.decision_function(x)

    def decision_gradient(self, x):
        """Return the gradient of the antimodel decision function evaluated at 
        x. x must be the raw feature vector, ie not normalized or transformed.
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
        pass

    def guess(self, constraints):
        """Take an initial guess at the feature vector under constraints"""
        r = [
            f["type"](uniform(f["min"], f["max"])) for f in self.feature_specs 
        ]
        for c in constraints:
            c["init"](r)
        return np.array(r)

    def run():
        pass
