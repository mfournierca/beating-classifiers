from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRANSFORM = ["scaler", StandardScaler()]


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
