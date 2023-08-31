
from basics.utilities import *
from sklearn.metrics.pairwise import cosine_similarity

def accuracy(dataset):
    return sum(tweet.GS==tweet.predicted for tweet in dataset)/len(dataset)

def getmetrics(scores, GS, show=True, tabular=False):
    try:
        scores = [tweet.predicted for tweet in scores.tweets]
    except:
        pass
    try:
        GS = [tweet.GS for tweet in GS.tweets]
    except:
        pass
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    macroF = 0
    n = 0
    for scorelist, gslist in zip(scores, GS):
        tpM =0
        fpM = 0
        fnM = 0
        for score, gs in zip(scorelist, gslist):
            if score == 1 and gs == 1:
                tp += 1
                tpM += 1
            elif score == 1:
                fp += 1
                fpM += 1
            elif gs == 1:
                fn += 1
                fnM += 1
            else:
                tn += 1
        if sum([tpM, fpM, fnM]) > 0:
            macroF += tpM/(tpM+0.5*(fpM+fnM))
            n += 1
    if n > 0:
        macroF = macroF/n
    if show:
        return showscores((macroF, tp, tn, fp, fn), tabular=tabular)
    else:
        return (macroF, tp, tn, fp, fn)

def showscores(scores, tabular=True):
    macroF, tp, fp, fn, tn = scores
    if tp > 0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f = tp/(tp+0.5*(fp+fn))
        j = tp/(tp+(fp+fn))
        a = (tp+tn)/(tp+tn+fp+fn)
    else:
        p = r = f = j = 0
    result = (p, r, f, macroF, j, a)
    if tabular:
        return "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%result
    else:
        return "P %.3f, R %.3f, micro F1 %.3f, macro F1 %.3f, Jaccard %.3f, accuracy %.3f"%result

def localmetrics(c):
    GS = c.test.GS
    scores = c.test.scores
    emotions = c.train.emotions
    if len(GS[0]) > len(emotions):
        emotions.append("--")
    GS = numpy.array(GS)
    scores = numpy.array(scores)
    for i, e in enumerate(emotions):
        tp = fp = fn = 0
        for g, s in zip(GS[:, i], scores[:, i]):
            if g == 1:
                if s == 1:
                    tp += 1
                else:
                    fn += 1
            elif s == 1:
                fp += 1
        try:
            p = tp/(tp+fp)
        except:
            p = 0
        try:
            r = tp/(tp+fn)
        except:
            r = 0
        try:
            f = tp/(tp+0.5*(fp+fn))
        except:
            f = 0
        print("%s: P %.3f R %.3f F %.3f (%s)"%(e, p, r, f, tp+fn))
        
"""
For every e that GS and predicted agree on
    c[e][e] +=1
    remove e from GS and predicted
There's now E = |GS'|+|predicted'] of error left: we want to apportion that
If GS' is empty 
    for e in predicted'
        c["--"][e] += 1
if predicted' is empty
    for e in GS':
        c[e]["--"] += 1
else
    How many GS did we not find? 
    Each of them could correspond to one of the predicted: 
    so give each c[g][p] 1/|p| of the blame. That's going to
    give you a total of |GS| blame. Seems sensible enough.
    for e in predicted'
       for f in GS'
          c[e][f] += 1/|GS'|

If GS["--"] == 1 and not predicted["--"] == 1 then every E in predicted is wrong, i.e. c["--"][E] += 1
If GS["
"""
def confusion(self, tweets=None):
    if isinstance(self, list):
        tweets = []
        for clsf in self:
            emotions = clsf.train.emotions
            tweets += clsf.test.tweets
    elif tweets is None:
        tweets = self.test.tweets
        emotions = self.train.emotions
    return confusionhelper(emotions, [(tweet.GS, tweet.predicted) for tweet in tweets])

def confusionhelper(emotions, pairs):
    c = {e: counter() for e in emotions}
    unassigned = "--"
    emptyset = set()
    for (gs, p) in pairs:
        """
        Set gs and p to the emotions assigned to this tweet in the Golf Standard and the predition:
        we use "--" for the cases where no emotion was assigned (note: this is *not* neutral, this is
        no emotion was assigned)
        """
        gs = set(emotions[i] if i < len(emotions) else "--" for i, e in enumerate(gs) if e == 1)
        p = set(emotions[i] if i < len(emotions) else "--" for i, e in enumerate(p) if e == 1)
        if gs == {"neutral"}:
            if p == emptyset:
                c["neutral"].add("neutral")
                gs = emptyset
        """
        For anything that gs and p both contain, increment the diagonal entry
        and remove it from gs and p
        """
        for i in gs.intersection(p):
            if not i in c:
                c[i] = counter()
            c[i].add(i)
            gs.remove(i)
            p.remove(i)
        """
        If there's nothing left in gs then everything in p is a false positive.
        Distribute the "amount of false positiveness" across everying in p
        """
        if gs == emptyset:
            for j in p:
                if not unassigned in c:
                    c[unassigned] = counter()
                c[unassigned].add(j, 1/len(p))
        elif p == emptyset:
            """
            If there's nothing left in p then everything in gs is a false negative.
            Distribute the "amount of false negativeness" across everying in gs
            """
            for j in gs:
                c[j].add(unassigned, 1/len(gs))
        else:
            """
            Get to here if there's something in gs and something in p.
            So there's genuine confusion: distribute the "amount of p"
            evenly across the entries in gs
            """
            for i in gs:
                for j in p:
                    c[i].add(j, 1/len(p))
    return c

def showconfusionhelper(conf, emotions, format="%s"):
    if "--" in conf:
        emotions = emotions+["--"]
    emotions1 = emotions
    print("\t%s"%('\t'.join(emotions)))
    transform = int if format == "%s" else identity
    for e in emotions:
        s = "\t".join(format%(transform(conf[e][f]) if f in conf[e] else 0) for f in emotions1)
        print("%s\t%s"%(e[:6], s))
    return conf

def showconfusion(clsf, conf=None, format="%s"):
    if conf is None:
        conf = confusion(clsf)
    transform = int if format == "%s" else identity
    try:
        emotions = clsf.train.emotions
    except:
        emotions = clsf[0].train.emotions
    showconfusionhelper(conf, emotions, format=format)

"""
Just count the number if tweets whose predicted/Gold Standard
includes each emotion IGNORING NEUTRAL AND UNASSIGNED
and normalise the result
"""

def proportions(clsfs, emotions, which=lambda x: x.predicted, ignore=["neutral", "--"]):
    conf = numpy.zeros(len(emotions))
    for clsf in clsfs:
        for t in clsf.test.tweets:
            for i, k in enumerate(which(t)):
                try:
                    if not emotions[i] in ignore:
                        if k == 1:
                            conf[i] += 1
                except:
                    pass
    try:
        return conf/sum(conf)
    except:
        return 0

"""
Do it for the prediction and the Gold Standard, calculate
the similarity between the two
"""
def proportionality(clsfs, ignore=["neutral", "---"]):
    emotions = clsfs[0].train.emotions
    predictedproportions = proportions(clsfs, emotions, ignore=ignore)
    gsproportions = proportions(clsfs, emotions, ignore=ignore, which=lambda x: x.GS)
    try:
        cs = cosine_similarity(predictedproportions.reshape(1, -1), gsproportions.reshape(1, -1))[0][0]
    except:
        cs = 0
    return ", ".join("%s: %.2f"%z for z in zip(emotions, predictedproportions) if not z[0] in ignore), ", ".join("%s: %.2f"%z for z in zip(emotions, gsproportions) if not z[0] in ignore), cs
        
        
    
