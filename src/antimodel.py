import numpy as np
import pandas as pd
from random import uniform

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
        self.antimodel = model.logistic()

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
            
    def guess(self, constraints):
        # get the antimodel parameters and calculate the gradient
        # minimize the gradient under constraints
        # get the resulting feature vector
        # predict and add to our training set
        # return the feature vector
        pass

    def run():
        pass
