import sys
sys.path.append("..")
from basics.utilities import *
from basics.datasets import WASSA, SEM4, SEM11, KWT, CARER, IMDB
from sklclassifiers import metrics, tweets
reload(tweets)
from basics import a2bw, corpora
import json
    
class BASECLASSIFIER():

    def readTrainingData(self, train, params={}):
        N = checkArg("N", params, 10000)
        useDF = False
        if isinstance(train, str):
            train = tweets.makeDATASET(train, params)
        train.tweets = train.tweets[:N]
        self.train = train
        self.matrix = tweets.tweets2sparse(self.train, N=N, params=params)
        """
        Convert the one-hot representation of the Gold Standard for each
        tweet to a class identifier
        """
        self.values = []
        for tweet in train.tweets[:N]:
            if checkArg("useonehot", params, default=True):
                self.values.append(tweets.onehot2value(tweet.GS, self.train.emotions))
            else:
                self.values.append(tweet.GS)
        """
        We need to make sure that there are no classes without any instances, or we will
        get into trouble when trying to do the multi-threshold versions of the standard
        algorithms
        """
        c = [0 for i in range(len(self.train.emotions))]
        for v in self.values:
            c[v] += 1
        # c is now a list of counts for each emotion: get the most frequent
        m = numpy.argmax(c)
        for i, x in enumerate(c):
            # Now look for columns that are empty
            if x == 0:
                # swap the first occurrence of the most frequent emotion for a zero one
                for j, y in enumerate(self.values):
                    if y == m:
                        self.values[j] = i
                        break
        
    def applyToTweets(self, testset, showprogress=False, threshold=None, probs=False):
        T = time.time()
        N = len(testset)
        l = []
        if threshold is None:
            threshold = self.params["threshold"]
        for i, tweet in enumerate(testset):
            if showprogress: progress(i, N, T, step=10)
            predicted = self.applyToTweet(tweet, threshold=threshold, probs=probs)
            tweet.predicted = predicted
            l.append(predicted)
        return l
    
    def chooseOther(self, token, d, topn=5):
        """
        If the classifier has a model, use that to find the 5 most similar
        words to the target and go through these looking for on        that is in the sentiment lexicon
        """
        if not token in d:
            try:
                for other in self.model.nearest(token, N=topn):
                    other = other[0]
                    if other in d:
                        return other
            except:
                pass
        return token
    
    def bestThreshold(self, bestthreshold, show=False, start=0, end=sys.maxsize):
        """
        Apply this classifier to the tweets we are interested in: setting
        probs=True forces it to return the values actually calculated by
        the classifier rather than the 0/1 version obtained by using the
        threshold
        """
        train = self.train.tweets[:len(self.test.tweets)]
        self.applyToTweets(train, threshold=0, probs=True)
        """
        The optimal threshold must lie somewhere between the smallest and
        largest versions scores for any tweet
        """
        if bestthreshold == "global":
            predicted = numpy.array([t.predicted for t in train])[start:end, :]
            lowest = threshold = numpy.min(predicted)
            highest = numpy.max(predicted)
            step = (highest-lowest)/20
            best = []
            GS = numpy.array([t.GS for t in train])[:, start:end]
            for i in range(20):
                l = self.applyToTweets(train, threshold=threshold)
                l = numpy.array(l)[:, start:end]
                """
                getmetrics returns macro F1, true positives, false positives, false negatives
                """
                m = metrics.getmetrics(GS, l, show=False)
                (macroF, tp, tn, fp, fn) = m
                j = tp/(tp+fp+fn)
                best = max(best, [j, threshold])
                if show:
                    print("%.2f %.3f"%(threshold, j))
                threshold += step
            return best[1]
        elif bestthreshold == "local":
            return numpy.array(list(self.bestThreshold("global", start=i, end=i+1) for i in range(len(self.train.emotions))))
        else:
            raise Exception("%s unexpected value for bestthreshold"%(bestthreshold))

    def bestLocalThreshold(self, i, show=True):
        start = i; end = i+1
        """
        Apply this classifier to the tweets we are interested in: setting
        probs=True forces it to return the values actually calculated by
        the classifier rather than the 0/1 version obtained by using the
        threshold
    """
        train = self.train.tweets[:len(self.test.tweets)]
        l = self.applyToTweets(train, threshold=0, probs=True)
        """
        The optimal threshold must lie somewhere between the smallest and
        largest versions for any tweet
    """
        predicted = numpy.array([t.predicted for t in train])[start:end, :]
        lowest = threshold = numpy.min(predicted)
        highest = numpy.max(predicted)
        step = (highest-lowest)/20
        best = []
        GS = numpy.array([t.GS for t in train])[:, start:end]
        for j in range(20):
            l = self.applyToTweets(train, threshold=threshold)
            l = numpy.array(l)[:, start:end]
            """
            getmetrics returns macro F1, true positives, false positives, false negatives
            """
            m = metrics.getmetrics(GS, l, show=False)
            (macroF, tp, tn, fp, fn) = m
            """ Jaccard """
            try:
                f1 = tp/(tp+fp+fn)
            except:
                f1 = 0
            best = max(best, [f1, threshold, tp, fn])
            threshold += step
        if show:
            other = self.bestThreshold("global", start=i, end=i+1)
            print("%s: %s, %s"%(self.train.emotions[i], best, other))
        return best[1]

    def getStrongLinks(self, emotion, N=10):
        probs = {}
        if checkArg("language", self, None) == "AR":
            convert = lambda word: a2bw.convert(word, a2bw.bw2atable)
        else:
            convert = identity
        for word in self.train.reducedindex:
            try:
                probs[convert(word)] = self.weight(word, emotion)
            except:
                pass
        return ["%s"%(k[0]) for k in sortTable(probs)[:N]]

def showWordProbs(self, words):
        if isinstance(words, str):
            words = words.split()
        print("\t%s"%("\t".join(self.train.emotions)))
        for word in words:
            if word in self.train.reducedindex:
                print("%s: %s"%(word, ", ".join("%.2f"%(self.weight(word, emotion)) for emotion in self.train.emotions)))           
           
def makeSplits(dataset, classifier, fold1=0, folds=sys.maxsize, params={}, showfolds=True):
    scorelist = []
    TEST = 0
    TRAIN = 0
    allfolds = []
    probs = checkArg("probs", params, default=False)
    N = checkArg("N", params, default=sys.maxsize)
    bestthreshold = checkArg("bestthreshold", params, default=False)
    for i in range(fold1, folds):
        if showfolds: print("FOLD %s"%(i))
        train, test = dataset.split(i, folds, N)
        T0 = time.time()
        clsf = classifier(train, params=params)
        clsf.test = test
        try:
            clsf.name = params["clsfname"]
        except:
            clsf.name = classifier
        if bestthreshold:
            params["threshold"] = clsf.threshold = clsf.bestThreshold(bestthreshold, show=False)
        T1 = time.time()
        predicted = clsf.applyToTweets(test.tweets, probs=probs)
        T2 = time.time()
        TEST += (T2-T1)/len(test.tweets)
        TRAIN += (T1-T0)
        clsf.traintime = TRAIN
        clsf.testtime = TEST
        score = metrics.getmetrics(predicted, test.GS, show=False)
        scorelist.append(score)
        clsf.score = score
        allfolds.append(clsf)
    S = len(scorelist)
    scores = numpy.zeros(len(scorelist[0]))
    for s in scorelist:
        scores += s
    scores[0] = scores[0]/S
    return allfolds, scores

"""
t = doItAll(paths=ENGLISH1[:1], params={"classifier": LEXCLASSIFIER, "threshold": 1, "N":70000, "stemmer": "none", "tokeniser": "SENT", "wthreshold": 5, "subclassifiers": None, "max_iter": 1000, "hiddenlayers": lambda x: (int(1.5*len(x.train.emotions), ), "useneutral": True, "maxdictsize": 10000, "useDF": False, "solver":"sgd", "alpha": 1e-5, "usesaved": True, "bestthreshold": "global"})
"""

def printparams(params):
    s = """params = {
"""
    for a in sorted(params.keys()):
        v = params[a]
        if isinstance(v, str):
            s += "%s: '%s',\n"%(a, v)
        elif isinstance(v, BASECLASSIFIER):
            s += "%s: %s,\n"%(a, v.__class__.__name__)
        else:
            s += "%s: %s,\n"%(a, v)
    print(s + "}")

def doItAll(f="wholething.csv", paths="CORPORA/TWEETS/SEM11/EN", params={}, end=False, N=sys.maxsize, showresults=False, name=None, showparams=False):
    clsftable = {}
    if isinstance(paths, str):
        paths = [paths]
    folds = checkArg("folds", params, default=0)
    fold1 = params["fold1"] = checkArg("fold1", params, default=0)
    N = params["N"]
    params["folds"] = folds
    if showparams:
        printparams(params)
    scorelist = []
    if len(paths) > 1:
        print("\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\tAccuracy")
    for path in paths:
        name = "%s-%s"%tuple(path.split("/")[-2:])
        params["language"] = name.split("-")[1]
        base = tweets.makeDATASET(os.path.join(path, f), params=params, N=N, showprogress=showparams)
        if folds == 0:
            if len(base.tweets[:N]) > 20000:
                folds = 5
            else:
                folds = 10
        folds, scores = makeSplits(base, params["classifier"], params=params, fold1=fold1, folds=folds, showfolds=showparams)
        score = numpy.array(list(c.score for c in folds))
        [mf1, tp, tn, fp, fn] = [sum(score[:, i]) for i in range(score.shape[1])]
        if showresults:
            s = "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%(params["dataset"].ljust(8, " "), tp/(tp+fp), tp/(tp+fn), tp/(tp+0.5*(fp+fn)), mf1/len(folds), tp/(tp+fp+fn), (tp+tn)/(tp+tn+fp+fn))
            print(s)
        clsftable[params["dataset"]] = folds
    return clsftable

"""
k = everything(classifiers=[DNNCLASSIFIER], datasets=ALL, overrides={"stemmer":"standard","bestthreshold":"global", "justone": True, "max_iter":2000, "N":70000, "useneutral": False, "usesaved": True})
"""

def everything(classifiers=[],
               datasets=[],
               overrides = {},
               allscores=None,
               showparams=True,
               showresults=True):
    params = {"threshold": 1, "N":70000, "stemmer": "none", "tokeniser": "standard", "wthreshold": 5, "max_iter": 1000, "hiddenlayers": lambda x: (int(1.5*len(x.train.emotions)),), "useneutral": False, "maxdictsize": 1000, "useDF": False, "solver":"sgd", "alpha": 1e-5, "usesaved": True, "bestthreshold": "global", "justone": False, "model": False,}
    if allscores is None:
        allscores = {}
    names = []
    if isinstance(overrides, dict):
        overrides = [overrides]
    if not isinstance(classifiers, list):
        classifiers = [classifiers]
    if not isinstance(datasets, list):
        datasets = [datasets]
    for clsf in classifiers:
        params["classifier"] = clsf
        clsfname = clsf.__name__[:-len("CLASSIFIER")]
        for dataset in datasets:
            for o, override in enumerate(overrides):
                for x in override:
                    params[x] = override[x]
                subclassifier = checkArg("subclassifiers", params, False)
                name = "%s-%s"%tuple(dataset.split("/")[-2:])
                params["dataset"] = name
                if subclassifier:
                    subclsfname = "%s-%s"%(clsfname, subclassifier.__name__[:-len("CLASSIFIER")])
                else:
                    subclsfname = clsfname
                if len(overrides) > 1:
                    subclsfname = "%s-%s"%(subclsfname, o)
                params["clsfname"] = subclsfname
                if not subclsfname in allscores:
                    allscores[subclsfname] = {}
                    names.append(subclsfname)
                allscores[subclsfname][name] = {}
                T0 = time.time()
                k = doItAll(paths=[dataset], params=params, name=name, showresults=showresults, showparams=showparams)
                print("TOTAL TIME TAKEN %s"%(time.time()-T0))
                clsflist = k[name]
                score = numpy.array(list(c.score for c in clsflist))
                (mf1, tp, tn, fp, fn) = [sum(score[:, i]) for i in range(score.shape[1])]
                p = tp/(tp+fp); r = tp/(tp+fn); f = tp/(tp+0.5*(fp+fn)); j = tp/(tp+fp+fn)
                allscores[subclsfname][name] = (tp/(tp+(fp+fn)/2), metrics.proportionality(clsflist)[2], clsflist, [p, r, f, mf1/len(clsflist), j])
        print(showlocalscores(allscores, subclsfname))
    print(showallscores(allscores))
    return allscores

def showallscores(allscores):
    clsfs = sorted(allscores.keys())
    out = """All scores for datasets %s using classifiers %s
\t%s
"""%(",".join(allscores[clsfs[0]]), ",".join(clsfs), "\t".join(clsfs))
    for dataset in allscores[clsfs[0]]:
        out += "%s"%(dataset)
        for clsf in clsfs:
            out += "\t%.3f"%(allscores[clsf][dataset][-1][-1])
        out += "\n"
    return out

def showlocalscores(allscores, clsf, proportionality=False):
    out = "Scores for %s\n"%(clsf)
    if proportionality:
        out += "\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\tProportionality\n"
    else:
        out += "\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\n"
    for dataset in allscores[clsf]:
        scores = allscores[clsf][dataset]
        if proportionality:
            out += "%s\t%s\n"%(dataset, "\t".join("%.3f"%(s) for s in scores[-1]+[scores[1]]))
        else:
            out += "%s\t%s\n"%(dataset, "\t".join("%.3f"%(s) for s in scores[-1]))
    return out

def makeClassifiers(datasets, classifierType):
    clsftable = {}
    ttable = {}
    for d in datasets:
        clsftable[d] = classifierType(tweets.makeDATASET(os.path.join(d, "train.csv"), stemmer=""))
        ttable[d] = tweets.makeDATASET(os.path.join(d, "dev.csv"), stemmer="")
    return clsftable, ttable

ENGLISH0 = [os.path.join(SEM4.PATH, "EN"), os.path.join(SEM11.PATH, "EN"), os.path.join(WASSA.PATH)]
CARER = [os.path.join(CARER.PATH)]
IMDB = [os.path.join(IMDB.PATH)]
ENGLISH = ENGLISH0+CARER+IMDB
IMAN = [os.path.join(KWT.PATH, "KWT.U", "AR"), os.path.join(KWT.PATH, "KWT.M", "AR")]
ARABIC = [os.path.join(SEM4.PATH, "AR"), os.path.join(SEM11.PATH, "AR")]+IMAN[1:]
SPANISH = [os.path.join(SEM4.PATH, "ES"), os.path.join(SEM11.PATH, "ES")]

MULTI = [os.path.join(SEM11.PATH, "EN"), os.path.join(SEM11.PATH, "AR"), os.path.join(SEM11.PATH, "ES")]+IMAN[1:]

FOREIGN = ARABIC+SPANISH
ALL = ENGLISH+FOREIGN

EXTRAS = IMAN[1:2]+IMDB

def timings(clsflist):
    clsf = clsflist[-1]
    print("training time %.3f sec (%s tweets/sec), testing time %.4f sec (%s tweets/sec)"%(clsf.traintime, int(len(clsf.train.tweets)/clsf.traintime), clsf.testtime, int(len(clsf.test.tweets)/clsf.testtime)))

def getemotions(self, scores, threshold=0):
    return ["%s:%.2f"%(e, s-sum(scores)/len(scores)) for e, s in zip(self.columns, scores) if s-sum(scores)/len(scores) > threshold]

def makeoverrides(start=1000, end=600000, step=1.5, override0=None, label="N"):
    overrides = []
    if override0 is None:
        override0 = {"folds": 5}
    N = start
    while N < end:
        override1 = {label: int(N)}
        for k in override0:
            override1[k] = override0[k]
        overrides.append(override1)
        N = N*step
    return overrides

def plot(clsf, dataset=ENGLISH[2], start=1000, end=600000, step=1.5, label="N", overrides=None):
    if overrides is None:
        overrides = makeoverrides(start=start, end=end, step=step, label=label)
    return everything(datasets=ENGLISH[-2], classifiers=clsf, overrides=overrides)

def makeplots(xxx, overrides, label="N"):
    print("training size\tJaccard\taccuracy\ttraining time\ttesting time")
    for i, k in enumerate(xxx.keys()):
        m = list(xxx[k].values())[0]
        j = m[-1][-1]
        f = m[-1][-3]
        c = m[2][-1]
        print("%s\t%s\t%s\t%s\t%.3f"%(overrides[i][label], j, f, int(c.traintime), c.testtime))
