from basics.utilities import *
import sklearn.svm
from sklearn.metrics.pairwise import cosine_similarity
import numpy

from sklclassifiers import baseclassifiers, tweets

from numpy import log, argmax

def safelog(threshold):
    return 0 if threshold == 0 else log(threshold)

class SKLEARNCLASSIFIER(baseclassifiers.BASECLASSIFIER):

    def applyToTweet(self, tweet1, threshold=None, probs=True):
        if threshold is None:
            threshold = checkArg("threshold", self.params, None)
        if isinstance(tweet1, str):
            tweet1 = tweets.makeTweet(tweet1, self.params)
        tweet = tweets.tweet2sparse(tweet1, self)
        scores = self.clsf.predict_proba(tweet)[0]
        if isinstance(scores, float) and len(self.train.emotions) == 2:
            scores = self.binaryScores(scores)
        if probs:
            return scores
        elif checkArg("justone", self.params, False):
            a = argmax(scores)
            scores = numpy.zeros(scores.shape[0])
            scores[a] = 1
        else:
            scores = (scores>threshold)*1
        return scores

    def binaryScores(self, scores):
        return [max(0, scores), min(0, scores)]
    
    def weight(self, word, emotion):
        try:
            j = self.train.reducedindex[word]
        except:
            return 0
        try:
            return self.weights[deindex(emotion, self.train.emotions), j]
        except:
            return 0

    def showAllWeights(self, words):
        print("\t%s"%("\t\t".join(self.train.emotions)))
        for word in words.split():
            print("%s:\t\t%s"%(word, "\t\t".join("%.3f"%(self.weight(word, emotion)) for emotion in self.train.emotions)))
