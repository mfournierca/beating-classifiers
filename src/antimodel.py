import numpy as np
import pandas as pd
from random import uniform

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src import model
from src import data


df = data.load_spambase()
SPAMBASE_FEATURE_SPECS = [
    {
        "name": c,
        "type": int if type(df[c].ix[0]) == np.int64 else float,
        "max": max(df[c]),
        "min": min(df[c])
    }
    for c in df.columns if c != "spam"
]


class AntiModel(object):
    
    def __init__(self, m):
        """A class which attacks a given model, looking for misclassification
        errors. 

        :param model: an sklearn model which is already trained
        :type model: object
        """
        self.model = m
        self._transform = FeatureUnion([("scaler", StandardScaler())])
        self.antimodel = Pipeline(
            steps=[
                ("transform", self._transform), 
                ("logistic", LogisticRegression())
            ]
        ) 
        self.prepare()

    def prepare(self, num_points=10000, feature_specs=SPAMBASE_FEATURE_SPECS):
        """Prepare the anti model. 

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
    
        # generate an initial set of random feature vectors
        rows = [
            {
                c["name"]: c["type"](uniform(c["min"], c["max"])) 
                for c in feature_specs
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
        x. 

        x must be the raw feature vector, ie not normalized or transformed."""
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
        # get the antimodel parameters and calculate the gradient
        # minimize the gradient under constraints
        # get the resulting feature vector
        # predict and add to our training set
        # return the feature vector
        pass

    def run():
        pass
