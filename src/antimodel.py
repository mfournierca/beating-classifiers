from src import model


class AntiModel(object):
    
    def __init__(model):
        """A class which attacks a given model, looking for misclassification
        errors. 

        :param model: an sklearn model which is already trained
        :type model: object
        """
        self.model = model
        self.antimodel = model.logistic()

    def prepare(initial_vectors=None):
        # generate an initial number of guesses of feature vectors
        # feed to the model to build a training set
        # fit the antimodel
        pass

    def guess(constraints):
        # get the antimodel parameters and calculate the gradient
        # minimize the gradient under constraints
        # get the resulting feature vector
        # predict and add to our training set
        # return the feature vector
        pass

    def run():
        pass
