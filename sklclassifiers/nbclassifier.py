from basics.utilities import *
from sklclassifiers import sklearnclassifier
from sklearn import naive_bayes

class NBCLASSIFIER(sklearnclassifier.SKLEARNCLASSIFIER):

    def __init__(self, train, params={}):
        # Convert the training data to sklearn format
        self.params = params
        self.threshold = checkArg("threshold", params, None)
        self.readTrainingData(train, params=params)
        # Make a naive bayes classifier
        self.clsf = naive_bayes.MultinomialNB()
        # Train it on the dataset
        self.clsf.fit(self.matrix, self.values)

    def weight(self, word, emotion):
        if word in self.train.reducedindex:
            probs = [numpy.exp(self.clsf.feature_log_prob_[i, :][self.train.reducedindex[word]]) for i in range(len(self.train.emotions))]
            return probs[deindex(emotion, self.train.emotions)]/sum(probs)
        else:
            return 0

