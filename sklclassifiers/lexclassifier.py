from basics.utilities import *
from math import log
import sklclassifiers.tweets
import sklclassifiers.metrics
from sklclassifiers.baseclassifiers import BASECLASSIFIER

class LEXCLASSIFIER(BASECLASSIFIER):

    def __init__(self, train, params={}):
        self.readTrainingData(train, params=params)
        self.threshold = params["threshold"]
        self.scoredict = {}
        self.params = params
        try:
            self.model = params["model"]
        except:
            self.model = None
        self.calculateScores()
    
    def applyToTweet(self, tweet, raw=False, threshold=None, probs=False):
        scores = numpy.zeros(len(self.train.emotions))
        if threshold is None:
            threshold = checkArg("threshold", self.params, None)
        if isinstance(tweet, str):
            tweet = tweets.makeTweet(tweet, self.params)
        for token in tweet.tokens:
            other = self.chooseOther(token, self.scoredict)
            if not token == other:
                print("%s chosen for %s: %s"%(other, token, ", ".join("%s:%.2f"%(emotion, score) for emotion, score in zip(self.train.emotions, self.scoredict[other]))))
            token = other
            if token in self.scoredict:
                for i, x in enumerate(self.scoredict[token]):
                    try:
                        scores[i] += x
                    except:
                        pass

        if sum(scores) > 0:
            scores = scores/max(scores)
        if probs:
            return scores
        elif checkArg("justone", self.params, default=False):
            p = numpy.argmax(scores)
            scores = numpy.zeros(len(scores))
            scores[p] = 1
            return scores
        else:
            """
            An emotion is selected if it scores nearly
            as highly as the best one.

            How do we extract the threshold from that?
            Normalise it so the maximum score is 1, use
            the threshold as now.
            """
            scores = (scores>=threshold)*1
            if "useneutral" in self.params and self.params["useneutral"]:
                if sum(scores[:-1]) > 0:
                    scores[-1] = 0
                else:
                    scores[-1] = 1
            return scores

    def weight(self, word, emotion):
        try:
            return self.scoredict[word][deindex(emotion, self.train.emotions)]
        except:
            raise Exception("unknown word: %s"%(word))

    def calculateScores(self):
        for tweet, gs in zip(self.train.tweets, self.train.GS):
            for word in tweet.tokens:
                if not word in self.scoredict:
                    self.scoredict[word] = numpy.zeros(len(tweet.GS))
                self.scoredict[word] += gs
        for word in self.scoredict:
            s = sum(self.scoredict[word])
            if s > 0:
                self.scoredict[word] = self.scoredict[word]/s

class CPCLASSIFIER(LEXCLASSIFIER):

    def calculateScores(self):
        for tweet, gs in zip(self.train.tweets, self.train.GS):
            for word in tweet.tokens:
                if not word in self.scoredict:
                    self.scoredict[word] = numpy.zeros(len(self.train.emotions))
                try:
                    self.scoredict[word] += gs
                except:
                    pass
        for word in self.scoredict:
            s = sum(self.scoredict[word])
            if s > 0:
                self.scoredict[word] = self.scoredict[word]/s
                m = numpy.mean(self.scoredict[word])
                for i, x in enumerate(self.scoredict[word]):
                    self.scoredict[word][i] = max(0, x-m)**1.5
