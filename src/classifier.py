from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

TRANSFORM = FeatureUnion([("scaler", StandardScaler())])


def naive_bayes():
    return Pipeline(
        steps=[
            ("transform", TRANSFORM),
            ("gnb", GaussianNB())
        ]
    )


def logistic():
    return Pipeline(
        steps=[
            ("transform", TRANSFORM),
            ("logistic", LogisticRegression())
        ]
    )


def svm():
    return Pipeline(
        steps=[
            ("transform", TRANSFORM),
            ("svm", SVC())
        ]
    )
