import logbook

from src import data, classifier, anticlassifier, features

MODELS = [
    classifier.logistic,
    classifier.svm
]
N = 1000


def evaluate(m):
    xtrain, xtest, ytrain, ytest = data.load_spambase_test_train()
    m.fit(xtrain, ytrain)
    a = anticlassifier.AntiClassifier(m, features.SPAMBASE_FEATURE_SPECS)
    
    score = m.score(xtest, ytest)
    logbook.info("model score: {0}".format(score))
    
        

def evaluate_all():
    pass
