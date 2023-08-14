from utilities import *
import metrics, tweets
importlib.reload(metrics)
importlib.reload(tweets)
import lexclassifier
importlib.reload(lexclassifier)
import a2bw
    
class BASECLASSIFIER():

    def readTrainingData(self, train, args={}):
        N = checkArg("N", args, 10000)
        useDF = False
        if isinstance(train, str):
            train = tweets.makeDATASET(train, args)
        train.tweets = train.tweets[:N]
        self.train = train
        self.matrix = tweets.tweets2sparse(self.train, N=N, args=args)
        """
        Convert the one-hot representation of the Gold Standard for each
        tweet to a class identifier
        """
        self.values = []
        for tweet in train.tweets[:N]:
            if checkArg("useonehot", args, default=True):
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
            threshold = self.args["threshold"]
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

"""
Simplified for book version
"""
                
def makeSplits(dataset, classifier, folds):
    scores = []
    N = len(dataset)/fold
    """
    Randomize the dataset, but make sure that you always shuffle it 
    the same way
    """
    random.seed(0)
    random.shuffle(pruned)
    for i in range(folds):
        # test is everything from i*N to (i+1)*N, train is everything else
        train, test = dataset[:i*N]+dataset[(i+1)*N:], dataset[i*N:(i+1)*N]
        clsf = classifier.train(training)
        score = clsf.test(test)
        scores.append(score)
    return scores
           
def makeSplits(dataset, classifier, S=10, S0=0, folds=sys.maxsize, args={}):
    scorelist = []
    TEST = 0
    TRAIN = 0
    N = args["N"]
    if len(dataset.tweets[:N]) > 20000:
        folds = min(folds, 5)
    allfolds = []
    probs = checkArg("probs", args, default=False)
    bestthreshold = checkArg("bestthreshold", args, default=False)
    for i in range(S0, min(S, folds)):
        print("FOLD %s"%(i))
        train, test = dataset.split(i, S, N)
        T0 = time.time()
        clsf = classifier(train, args=args)
        clsf.test = test
        try:
            clsf.name = args["clsfname"]
        except:
            clsf.name = classifier
        if bestthreshold:
            args["threshold"] = clsf.threshold = clsf.bestThreshold(bestthreshold, show=False)
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
t = doItAll(paths=ENGLISH1[:1], args={"classifier": LEXCLASSIFIER, "threshold": 1, "N":70000, "stemmer": "none", "tokeniser": "SENT", "wthreshold": 5, "subclassifiers": None, "max_iter": 1000, "hiddenlayers": lambda x: (int(1.5*x), ), "useneutral": True, "maxdictsize": 10000, "useDF": False, "solver":"sgd", "alpha": 1e-5, "usesaved": True, "bestthreshold": "global"})
"""

def doItAll(f="wholething.csv", paths="CORPORA/TWEETS/SEM11/EN", args={}, end=False, N=sys.maxsize, show=True, name=None):
    clsftable = {}
    # args['classifier'] = classifier.__name__
    print(", ".join("%s: %s"%(k, args[k]) for k in args))
    if isinstance(paths, str):
        paths = [paths]
    folds = checkArg("folds", args, default=sys.maxsize)
    S0 = checkArg("S0", args, default=0)
    scorelist = []
    if len(paths) > 1:
        print("\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\tAccuracy")
    for path in paths:
        name = "%s-%s"%tuple(path.split("/")[-2:])
        args["language"] = name.split("-")[1]
        args["name"] = name
        base = tweets.makeDATASET(os.path.join(path, f), args=args, N=N, showprogress=False)
        folds, scores = makeSplits(base, args["classifier"], args=args, S0=S0, folds=folds)
        score = numpy.array(list(c.score for c in folds))
        [mf1, tp, tn, fp, fn] = [sum(score[:, i]) for i in range(score.shape[1])]
        if show:
            s = "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%(name.ljust(8, " "), tp/(tp+fp), tp/(tp+fn), tp/(tp+0.5*(fp+fn)), mf1/len(folds), tp/(tp+fp+fn), (tp+tn)/(tp+tn+fp+fn))
            print(s)
        clsftable[name] = folds
    print(", ".join("%s: %s"%(k, args[k]) for k in args))
    return clsftable

"""
k = everything(classifiers=[DNNCLASSIFIER], datasets=ALL, overrides={"stemmer":"standard","bestthreshold":"global", "justone": True, "max_iter":2000, "N":70000, "useneutral": False, "usesaved": True})
"""

def everything(classifiers=[],
               datasets=[],
               overrides = {},
               allscores=None):
    args = {"threshold": 1, "N":70000, "stemmer": "none", "tokeniser": "standard", "wthreshold": 5, "max_iter": 20000, "hiddenlayers": lambda x: (int(1.5*x),), "useneutral": True, "maxdictsize": 10000, "useDF": False, "solver":"sgd", "alpha": 1e-5, "usesaved": True, "bestthreshold": "global", "justone": False, "model": False}
    if allscores is None:
        allscores = {}
    names = []
    if isinstance(overrides, dict):
        overrides = [overrides]
    if not isinstance(classifiers, list):
        classifiers = [classifiers]
    for clsf in classifiers:
        args["classifier"] = clsf
        clsfname = clsf.__name__[:-len("CLASSIFIER")]
        for dataset in datasets:
            for o, override in enumerate(overrides):
                for x in override:
                    args[x] = override[x]
                subclassifier = checkArg("subclassifiers", args, False)
                name = "%s-%s"%tuple(dataset.split("/")[-2:])
                args["dataset"] = name
                if subclassifier:
                    subclsfname = "%s-%s"%(clsfname, subclassifier.__name__[:-len("CLASSIFIER")])
                else:
                    subclsfname = clsfname
                args["clsfname"] = subclsfname
                if not subclsfname in allscores:
                    allscores[subclsfname] = {}
                    names.append(subclsfname)
                allscores[subclsfname][name] = {}
                k = doItAll(paths=[dataset], args=args, name=name)
                clsflist = k[name]
                score = numpy.array(list(c.score for c in clsflist))
                (mf1, tp, tn, fp, fn) = [sum(score[:, i]) for i in range(score.shape[1])]
                p = tp/(tp+fp); r = tp/(tp+fn); f = tp/(tp+0.5*(fp+fn)); j = tp/(tp+fp+fn)
                allscores[subclsfname][name] = (tp/(tp+(fp+fn)/2), metrics.proportionality(clsflist)[2], clsflist, [p, r, f, mf1/len(clsflist), j])
                print(showlocalscores(allscores, proportionality=True))
    return allscores

def showallscores(allscores, clsfs=None):
    if clsfs is None:
        clsfs = sorted(allscores.keys())
    else:
        clsfs = [clsf for clsf in clsfs if clsf in allscores]
    out = "\t"+"\t".join(clsfs)+"\n"
    for dataset in list(allscores[clsfs[0]]):
        out += "%s\t%s\n"%(dataset, [[allscores[clsf][dataset][0]]+ allscores[clsf][dataset][-1] if clsf in allscores and dataset in allscores[clsf] else (0, 0) for clsf in clsfs])
    return out

def showlocalscores(allscores, clsfs=None, proportionality=True):
    if clsfs is None:
        clsfs = sorted(allscores.keys())
    else:
        clsfs = [clsf for clsf in clsfs if clsf in allscores]
    out = "\t"+"\t".join(clsfs)+"\n"
    if proportionality:
        out += "\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\tProportionality\n"
    else:
        out += "\tPrecision\tRecall\tmicro F1\tmacro F1\tJaccard\n"
    for dataset in list(allscores[clsfs[0]]):
        scores = allscores[clsfs[0]][dataset]
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

def applyClassifiers(clsftable, ttable, threshold=0.5):
    print("\t\tPrecision\tRecall\t\tmicro F1\tmacro F1\tJaccard")
    for d in clsftable:
        clsf = clsftable[d]
        test = ttable[d]
        try:
            d = "-".join(d.split("/")[-2:])
        except:
            pass
        clsf.threshold = threshold
        print("%s\t%s"%(d, metrics.metrics(clsf.applyToTweets(test.tweets), test.GS, tabular=(len(clsftable) > 1))))

def getemotions(self, scores, threshold=0):
    return ["%s:%.2f"%(e, s-sum(scores)/len(scores)) for e, s in zip(self.columns, scores) if s-sum(scores)/len(scores) > threshold]

ENGLISH0 = ["CORPORA/TWEETS/SEM4/EN", "CORPORA/TWEETS/SEM11/EN", "CORPORA/TWEETS/WASSA/EN"]
CARER = ["CORPORA/TWEETS/CARER/EN"]
IMDB = ["CORPORA/TWEETS/IMDB/EN"]
ENGLISH = ENGLISH0+CARER+IMDB
IMAN = ["CORPORA/TWEETS/IMAN/KWT.U/AR", "CORPORA/TWEETS/IMAN/KWT.M/AR"]
ARABIC = ["CORPORA/TWEETS/SEM4/AR", "CORPORA/TWEETS/SEM11/AR"]+IMAN[1:]
SPANISH = ["CORPORA/TWEETS/SEM4/ES", "CORPORA/TWEETS/SEM11/ES"]

MULTI = ["CORPORA/TWEETS/SEM11/EN", "CORPORA/TWEETS/SEM11/AR", "CORPORA/TWEETS/SEM11/ES"]+IMAN[1:]

FOREIGN = ARABIC+SPANISH
ALL = ENGLISH+FOREIGN

EXTRAS = IMAN[1:2]+IMDB

def timings(train, test, classifier, step=1.5):
    N = 5000
    print("training size\tJaccard\ttraining time")
    while N < step*len(train.tweets):
        T0 = time.time()
        clsf = classifier(train, N=int(N))
        T1 = time.time()
        (p, r, f, macroF, j) = metrics.metrics(clsf.applyToTweets(test.tweets), test.GS, show=False)
        print("%s\t%.3f\t%.1f"%(min(int(N), len(train.tweets)), j, T1-T0))
        N = N*step

samplewords0 = "adores angry happy hate irritated joy love sad scared sorrow terrified"
samplewords1 = "a and the"
