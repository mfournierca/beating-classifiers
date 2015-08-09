import logbook
import pandas as pd
from src import classifier, anticlassifier
from src.features import (
    xtrain,
    ytrain,
    xtest,
    ytest,
    SPAMBASE_FEATURE_SPECS,
    create_greater_than_constraint
)

from sklearn.feature_selection import f_classif

MODELS = [
    classifier.logistic,
    classifier.svm
]
N = 1000
I = 10


def most_significant_features(x, y, limit=10):
    """Use an ANOVA hypothesis test to return the column names of the features
    in x that have the lowest p-value.

    We use the lowest p-value to indicate the features in x that are most
    significant in predicting y"""

    anova_p_values = f_classif(x, y)[1]
    sig = [(index, p_value) for index, p_value in enumerate(anova_p_values)]
    sig.sort(key=lambda x: x[1])
    sig = sig[:limit]
    columns = x.columns
    return [{"index": x[0], "name": columns[x[0]]} for x in sig]


def anticlassifier_precision(classifier, feature_specs, constraints, x, y):
    anti = anticlassifier.AntiClassifier(classifier, feature_specs)
    record = pd.DataFrame(
        columns=[i["name"] for i in feature_specs] + ["classifier_predict"]
    )
    for i in range(N):
        f = anti.get(constraints)
        p = classifier.predict(f)
        record.loc[len(record) + 1] = list(f) + [p]

    # we want the generated feature vectors to get through the classifier,
    # i.e. success is when the classifier predicts 0
    precision = (
        1.0 -
        float(sum(record["classifier_predict"])) /
        len(record["classifier_predict"])
    )
    return precision, record


def evaluate(classifier, constrain="max"):
    """Evaluate the performance of the anticlassifier against the given
    classifier.
    """
    classifier.fit(xtrain, ytrain)
    score = classifier.score(xtest, ytest)

    df = pd.DataFrame(
        columns=["significant_features_constrained", "anticlassifier_score"])
    constraints = []
    feature_specs = SPAMBASE_FEATURE_SPECS

    # base case
    p, r = anticlassifier_precision(
        classifier, feature_specs, constraints, xtest, ytest
    )
    df.loc[len(df) + 1] = [0, p]

    # constrain each of the significant features and test classifier precision
    significant = most_significant_features(xtest, ytest)
    for index, sig in enumerate(significant):
        spec = [f for f in feature_specs if f["name"] == sig["name"]]
        assert len(spec) == 1
        spec = spec[0]

        if constrain == "max":
            constraint_value = int(spec["max"])
        elif constrain == "min":
            constraint_value = int(spec["min"])
        elif constrain == "mid":
            constraint_value = int((spec["min"] + spec["max"]) / 2)
        else:
            raise ValueError("constrain but but one of min, max, mid")

        constraint = create_greater_than_constraint(
            xtrain,
            sig["name"],
            sig["index"],
            constraint_value,
            constraint_value
        )

        constraints.append(constraint)
        precision, record = anticlassifier_precision(
            classifier, feature_specs, constraints, xtest, ytest
        )
        df.loc[len(df) + 1] = [index + 1, precision]

    df["classifier_score"] = score
    return df

