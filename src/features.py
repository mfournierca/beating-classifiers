import numpy as np
from random import randint
from src import data

xtrain, xtest, ytrain, ytest = data.load_spambase_test_train()


# In the final version of this module we shouldn't be tightly coupled to the
# features of the classifier that we are attacking. The decoupling would
# require an invertible mapping between feature vectors and objects (e.g. email
# messages) which would make matching feature column names and order
# irrelevant. We don't have that mapping yet so we rely on our features
# having the same order and names as the classifier we attack.
SPAMBASE_FEATURE_SPECS = [
    {
        "name": c,
        "type": int if type(xtrain[c].ix[0]) == np.int64 else float,
        "max": max(xtrain[c]),
        "min": min(xtrain[c])
    }
    for c in xtrain.columns if c != "spam"
]


def create_greater_than_constraint(
        x,
        column_name,
        column_index,
        greater_than,
        upper_bound
    ):
    """Create a constraint that requires a certain feature to be greater than
    a specified value.

    The feature constraints are used by the anticlassifier when generating
    vectors. The constraints are used in two roles:

    - when generating the initial guess of the feature vector
    - in the scipy.optimize routines that seek the optimal vector

    These roles require the constraints to be expressed in a certain format.

    The initial guess generation requires:

    - That the "init" function be a callable

    See anticlassifier.py for details.

    The scipy.optimize routines require:

    - the constraints must be a list of dictionary
    - the "type" field must be set on each dictionary
    - the "fun" field on each dictionary must be a callable

    For details, refer to:

    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    assert x[column_index] == column_name
    return {
        "name": "{0}_gt_{1}".format(column_name, greater_than),
        "type": "ineq",
        "fun": lambda x: x[column_index] - greater_than,
        "init": lambda x: x.__setitem__(
            column_index, randint(greater_than, upper_bound))
    }


# the constraints to pass onto the minimizer.
# depend on the order of the feature specs
SPAMBASE_CONSTRAINTS = []
SPAMBASE_CONSTRAINTS.append(
    create_greater_than_constraint(
        xtrain,
        "capital_run_length_average",
        0,
        10,
        25
    )
)
SPAMBASE_CONSTRAINTS.append(
    create_greater_than_constraint(
        xtrain,
        "capital_run_length_longest",
        1,
        5,
        10
    )
)

