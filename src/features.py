import numpy as np
from random import randint
from src import data


# In the final version of this module we won't be tightly coupled to the 
# features of the classifier that we are attacking. The decoupling would require
# a invertible mapping between feature vectors and objects (e.g. email 
# messages) which would make matching feature column names and order 
# irrelevant. We don't have that mapping yet so we rely on our features 
# having the same order and names as the classifier we attack.
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

# the constraints to pass onto the minimizer. 
# depend on the order of the feature specs
SPAMBASE_CONSTRAINTS = []

assert xtrain.columns[0] == "capital_run_length_average"
SPAMBASE_CONSTRAINTS.append({
    "name": "capital_run_length_average_gt_10",
    "type": "ineq",
    "fun": lambda x: x[0] - 10,
    "init": lambda x: x.__setitem__(0, randint(10, 25))
})

assert xtrain.columns[1] == "capital_run_length_longest"
SPAMBASE_CONSTRAINTS.append({
    "name": "capital_run_length_longest_gt_5",
    "type": "ineq",
    "fun": lambda x: x[1] - 5,
    "init": lambda x: x.__setitem__(1, randint(5, 10))
})

