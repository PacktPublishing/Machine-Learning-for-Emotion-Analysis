from basics.utilities import *
import sklearn.svm
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from sklclassifiers import tweets, sklearnclassifier

class MULTICLASSIFIER(sklearnclassifier.SKLEARNCLASSIFIER):

    def __init__(self, train, showprogress=True, params={}):
        self.train = train
        T = time.time()
        self.datasets = {}
        self.classifiers = {}
        self.params = params
        self.train.reducedindex = {}
        # Find out what kind of classifier to use for the indiviual emotions
        subclassifier = params["subclassifiers"]
        for i in range(len(self.train.emotions)):
            print("TRAINING %s"%(i))
            squeezed = self.squeeze(i)
            if squeezed:
                self.datasets[i] = squeezed
                self.classifiers[i] = subclassifier(self.datasets[i], params=params)
                for w in self.classifiers[i].train.reducedindex:
                    if not w in self.train.reducedindex:
                        self.train.reducedindex[w] = len(self.train.reducedindex)

            """
            Collapse the Gold Standard for each tweet so that we just
            have two columns, one for emotion[i] in the original and one
            for the rest. Using numpy.sign(sum(gs[:i]+gs[i+1:])) will
            set the second column to be 0 if all the columns except i
            are 0, otherwise it sets it to 1. *It is possible for both
            the resulting columns to be 0 and for both to be 1*
            Everything else in here is just about copying the original
            """
        print("ALL TRAINED")
            
    def squeeze(self, i):
        l = []
        for tweet in self.train.tweets:
            gs = tweet.GS
            gs0 = gs[i]
            gs1 = 1 if gs0 == 0 else 0
            scores=[gs[i], numpy.sign(sum(gs[:i])+sum(gs[i+1:]))]
            tweet = tweets.TWEET(id=tweet.id, tf=tweet.tf, scores=scores, tokens=tweet.tokens)
            l.append(tweet)
        emotion = self.train.emotions[i]
        emotions = [emotion, "not %s"%(emotion)]
        return tweets.DATASET(emotions, l, self.train.df, self.train.idf, self.train.params)

    def applyToTweet(self, tweet, threshold=None, probs=True):
        k = [0 for i in self.train.emotions]
        for i in self.classifiers:
            c = self.classifiers[i]
            p = numpy.argmax(c.applyToTweet(tweet, threshold=threshold, probs=probs))
            """
            if classifier i says that this tweet expresses
            the classifier's emotion (i.e. if the underlying
            classifier returns 0, meaning that the classifier chose X rather than not-X) 
            then set the ith column of the
            main classifier to 1
            """
            if p == 0:
                k[i] = 1
        return k
    
    def weight(self, word, emotion):
        for clsf in self.classifiers:
            try:
                return self.classifiers[clsf].weight(word, emotion)
            except:
                pass
        return -1
