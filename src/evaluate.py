import logbook
import pandas as pd
from copy import deepcopy
from src import classifier, anticlassifier, features
from src.features import xtrain, ytrain, xtest, ytest, SPAMBASE_FEATURE_SPECS

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
    return [(x[0], columns[x[0]]) for x in sig]


def anticlassifier_precision(classifier, feature_specs, constraints, x, y):
    anti = anticlassifier.AntiClassifier(classifier, features)
    record = pd.DataFrame(
        columns=[i["name"] for i in feature_specs] + ["classifier_predict"]
    )
    for i in range(N):
        f = anti.get(constraints)
        p = classifier.predict(f)
        record.append(f + [p])

    precision = (
        float(sum(record["classifier_predict"])) /
        len(record["classifier_predict"])
    )
    return precision, record


def evaluate(classifier):
    """Evaluate the performance of the anticlassifier against the give
    classifier.
    """
    classifier.fit(xtrain, ytrain)
    score = classifier.score(xtest, ytest)
    logbook.info("classifier score: {0}".format(score))

    df = pd.DataFrame(
        colums=["significant_features_restricted", "precision"])
    constraints = []

    # base case
    p, r = anticlassifier_precision(
        classifier, SPAMBASE_FEATURE_SPECS, constraints, xtest, ytest
    )
    df.append([0, p])

    significant = most_significant_features(xtest, ytest)
    for name in significant:
        pass

    df["classifier_score"] = pd.Series(score)
    return df
