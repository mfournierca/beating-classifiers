import logbook
import pandas as pd
from src import data, classifier, anticlassifier, features

MODELS = [
    classifier.logistic,
    classifier.svm
]
N = 1000
I = 10


def evaluate(m):
    xtrain, xtest, ytrain, ytest = data.load_spambase_test_train()
    m.fit(xtrain, ytrain)
    a = anticlassifier.AntiClassifier(m, features.SPAMBASE_FEATURE_SPECS)
    
    score = m.score(xtest, ytest)
    logbook.info("classifier score: {0}".format(score))
   
    record = pd.DataFrame(
        columns=[i["name"] for i in features.SPAMBASE_FEATURE_SPECS] + 
                ["classifier_predict"]
    )
    for i in range(N): 
        f = a.get(features.SPAMBASE_CONSTRAINTS)
        p = m.predict(f)
        record.append(f + [p]) 
    return record        

 
def evaluate_all():
    pass
