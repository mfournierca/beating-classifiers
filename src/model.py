from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def logistic():
    steps = [
        ("logistic", LogisticRegression)    
    ]
    pipe = Pipeline(steps=steps)
    return pipe

