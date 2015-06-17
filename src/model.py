from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic():
    steps = [
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression())    
    ]
    pipe = Pipeline(steps=steps)
    return pipe

