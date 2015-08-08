"""Import this module to automatically prepare data and classifiers"""

import data
import classifier

xtrain, xtest, ytrain, ytest = data.load_spambase_test_train()
logistic_model = classifier.logistic().fit(xtrain, ytrain)
svm_model = classifier.svm().fit(xtrain, ytrain)

