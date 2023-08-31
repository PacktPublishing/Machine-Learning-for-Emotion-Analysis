from basics.utilities import *
from chapter4 import arabicstemmer, spanishstemmer, stem3, tokenisers
reload(stem3)
from basics import corpora, a2bw
import numpy
import random
from scipy import sparse
import dill as pickle

languagePattern = re.compile(".*/(?P<language>[A-Z]*)/.*")

class TWEET:

    # id and params are for bookkeeping
    def __init__(self, id=False, tf=False, scores=False, tokens=False, params={}, src=None):
        self.id = id
        self.GS = scores
        self.tokens = tokens
        self.tf = normalise(tf)
        self.params = params
        self.src = src
        
    def __str__(self):
        return " ".join(self.tokens)

"""
Enumerate the list of values: as soon as you hit a 1, return the
position where you found it -- since it's a one-hot encoding there
will only be one such position.

If allowZeros is set to true then we will return a column *beyond* the
actual values for empty cases. If there is more than one non-zero column
then we will just return the first -- it's expected to be one-hot, so
using it with something else will give unexpected results
"""
def onehot2value(l, allowZeros=True):
    try:
        for i, x in enumerate(l):
            if x == 1:
                return i
        if allowZeros:
            return len(l)-1
        else:
            raise Exception("No non-zero value found")
    except Exception as e:
        print("onehot2value failed: %s"%(l))
        raise e

def value2onehot(v, emotions):
    return [1 if e == v else 0 for e in emotions]


class DATASET:

    def __init__(self, emotions, tweets, df, idf, params={}, N=sys.maxsize):
        self.emotions = emotions
        self.tweets = tweets
        self.GS = [tweet.GS for tweet in self.tweets][:N]
        self.df = df
        self.idf = idf
        self.index = self.makeIndex()
        self.invindex = {v: x for x, v in self.index.items()}
        self.params = params
        
    def __iter__(self):
        for tweet in self.tweets:
            yield tweet

    def __getitem__(self, i):
        return self.tweets[i]

    def makeIndex(self):
        index = {}
        for tweet in self.tweets:
            for token in tweet.tokens:
                if not token in index:
                    index[token] = len(index)
        return index

    def split(self, n, S, N):
        tweets = self.tweets[:N]
        s = min(int(len(tweets)/S), 1000)
        return (makeSplit(tweets[:s*n]+tweets[s*(n+1):], self.emotions, self.params), makeSplit(tweets[s*n:s*(n+1)], self.emotions, self.params))
 
def makeTweet(tweet, params):
    try:
        tss = tweet.strip().split("\t")
        scores = [int(score) for score in tss[2:]]
        if checkArg("useneutral", params, False):
            scores.append(0)
            if not 1 in scores:
                scores[-1] = 1
        tweet, text, scores = tss[0], tss[1], numpy.array(scores)
    except:
        tweet, text, scores = "DUMMY", tweet, False
    d = {}
    if params["language"] == "AR":
        text = a2bw.convert(text, a2bw.a2bwtable)
        if params["tokeniser"] == "standard":
            tokens = tokenisers.tokenise(text, tokenisePattern=tokenisers.ARABICPATTERN)
        else:
            tokens = text.split()
        if params["stemmer"] == "standard":
            tokens = arabicstemmer.stemAll(tokens, arabicstemmer.TWEETGROUPS)
    elif params["language"] == "ES":
        if params["stemmer"] == "standard":
            tokens = spanishstemmer.stemAll(text)
        else:
            tokens = text.split()
    else:
        if params["tokeniser"] == "standard":
            tokens = tokenisers.tokenise(text.strip(), tokenisers.ENGLISHPATTERN)
        elif params["tokeniser"] == "standard1":
            tokens = tokenisers.tokenise(text.strip(), tokenisers.ENGLISHPATTERN1)
        elif params["tokeniser"] == "NLTK":
            tokens = tokenisers.NLTKtokenise(text.strip())
        else:
            tokens = text.strip().split()
        if params["stemmer"].startswith("standard"):
            # Don't actually want to strip off prefixes for sentiment classification
            stem3.PREFIXES = {}
            tokens = [stem3.stemAll(token, stem3) for token in tokens]
        elif params["stemmer"].startswith("morphy"):
            tokens = [stem3.morphyroot(token) for token in tokens]
    for w in tokens:
        try:
            d[w] += 1
        except:
            d[w] = 1
    return TWEET(id=tweet, tf=d, scores=scores, tokens=tokens, params=params, src=text)

def makeDATASET(src, params, N=sys.maxsize, showprogress=True, save=True):
    threshold = checkArg("threshold", params, 10)
    stemmer= params["stemmer"] = checkArg("stemmer", params, "none")
    tokeniser = params["tokeniser"] = checkArg("tokeniser", params, "none")
    f = open(src)
    dataset = list(f.read().split("\n"))[:int(N)]
    f.close()
    T0 = time.time()
    emotions = None
    tweets = []
    if showprogress:
        print("MAKING DATASET")
    for i, tweet in enumerate(dataset):
        if showprogress: progress(i, len(dataset), T0, step=100)
        if tweet == "":
            continue
        if emotions is None:
            emotions = tweet.split()[2:]
            if checkArg("useneutral", params, False):
                emotions.append("neutral")
        else:
            tweets.append(makeTweet(tweet, params=params))
    T1 = time.time()
    if showprogress:
        print("\nDONE -- %.2f seconds"%(T1-T0))
    pruned = prune(tweets)
    random.seed(0)
    random.shuffle(pruned)
    dataset = makeSplit(pruned, emotions, params=params)
    dataset.src = src
    if save:
        try:
            picklefile = open(picklefile, "wb")
            pickle.dump([params, dataset], picklefile)
            picklefile.close()
        except Exception as e:
            pass
    return dataset

def getDF(documents):
    # adding something to df either sets or increments a counter
    df = counter()
    for document in documents:
        if isinstance(document, str):
            document = document.split()
        # for each unique word in the document increment df
        for w in set(document):
            df.add(w)
    
    idf = {}
    for w in df:
        idf[w] = 1.0/float(df[w]+1)
    return df, idf

def makeSplit(tweets, emotions, params=None):
    df = counter()
    index = {}
    for i, tweet in enumerate(tweets):
        for w in tweet.tokens:
            df.add(w)
    """
    remove singletons from the idf count
    """
    idf = {}
    for w in list(df.keys()):
        idf[w] = 1.0/float(df[w]+1)
    return DATASET(emotions, tweets, df, idf, params=params)

def sentence2vector(sentence, index, convert2sparse=True, idf={}):
    vector = numpy.zeros(len(index))
    if isinstance(sentence, str):
        sentence = sentence.split()
    for word in sentence:
        if word in idf:
            inc = idf[word]
        else:
            inc = 1
        print(word, index[word], inc)
        vector[index[word]] += inc
    if convert2sparse:
        vector = sparse.csr_matrix(vector)
    return vector

def cooccurrences(sentences, df, idf, w=3):
    c = {}
    for sentence in sentences:
        if isinstance(sentence, str):
            sentence = sentence.split()
        for i, word in enumerate(sentence):
            if not word in c:
                c[word] = counter()
            for other in sentence[i-w:i+w+1]:
                if not other == word and other in df and df[other] > 20:
                    c[word].add(other, 1)
    
    return c

def tweet2sparse(tweet, train, useDF=False):
    try:
        train = train.train
    except:
        pass
    try:
        tweet = tweet.tokens
    except:
        pass
    """
    Set up a large empty array to hold the various values
    """
    s = numpy.zeros(len(train.reducedindex))
    """
    Find the sum of the idf scores for the words in the tweet (to be used
    if we want to use tf-idf to weight the values of the tokens)
    """
    t = sum(train.idf[token] for token in tweet if token in train.idf)
    for token in tweet:
        """
        For each token, set the value of the dimension that it corresponds to.
        Because we will be using this for tweets in the training set AND in 
        the test set, we have to check that the token is indeed in the index,
        because otherwise it will not correspond to a dimension in the vector.
        The try-expect deals with this
        """
        try:
            if useDF:
                s[train.reducedindex[token]] = train.idf[token]/t
            else:
                s[train.reducedindex[token]] = 1
        except:
            pass
    """
    return this as a list containing a single normal array
    """
    return [s]

"""
Make a sparse matrix out of all the tweets in a training set (quicker
and cleaner than converting them one at a time and then putting them in
a single big matrix at the end)

To make a sparse matrix you collect parallel lists of rows, columns and data
for al cases where the data is non-zero. So what we have to do is go through
the tweets one by one (tweet number = row number), and then go through the
tokens in the tweet; we look the token up in the index (token index = column 
number), work out what value we want to use for that token (either 1 or its
idf) and add those to rows, columns, data. 

Then at the end you just make a sparse matrix out of all of that, plus a specification of the shape
"""
def tweets2sparse(train, N=sys.maxsize, params={"useDF": False, "wthreshold":1, "maxdictsize": 5000}):
    rows = []
    data = []
    columns = []
    train.reducedindex = {}
    for token, value in sortTable(train.df)[:params["maxdictsize"]]:
        if train.df[token] > params["wthreshold"] and not token in train.reducedindex:
            train.reducedindex[token] = len(train.reducedindex)
    # print("original word count %s, reduced wordcount %s"%(len(train.index), len(train.reducedindex)))
    for i, tweet in enumerate(train.tweets[:N]):
        t = sum(train.idf[token] for token in tweet.tokens)
        for token in tweet.tokens:
            if token in train.reducedindex:
                rows.append(i)
                columns.append(train.reducedindex[token])
                if params["useDF"]:
                    s = train.idf[token]/t
                else:
                    s = 1
                data.append(s)
    return sparse.csc_matrix((data, (rows, columns)), (len(train.tweets[:N]), len(train.reducedindex)))

def getemotions(self, scores, threshold=0):
    return [e for e, s in zip(self.emotions, scores) if s > threshold]

def showWords(words, lexlist):
    print("\t%s"%("\t".join(lexlist)))
    for w in words:
        print("%s\t%s"%(w, "\t".join(",".join(getemotions(lexlist[d], lexlist[d].scoredict[w])) for d in lexlist)))

def gethistogram(train):
    h = numpy.zeros(len(train.emotions))
    for g in train.GS[:len(train.tweets)]:
        h += g
    return ", ".join(["%s: %s"%(e, int(x)) for e, x in zip(train.emotions, h)])

def counts(train):
    counts = numpy.zeros(len(train.emotions), dtype=numpy.int)
    for tweet in train.tweets:
        counts[sum(tweet.GS)] += 1
    return counts

def lowesttoken(tweet, wordcount):
    l = None
    lscore = sys.maxsize
    for token in tweet.tokens:
        c = wordcount[token]
        if c < lscore:
            lscore = c
            l = token
    return l

def prune(tweets):
    pruned = []
    wordcount = counter()
    for tweet in tweets:
        for token in tweet.tokens:
            wordcount.add(token)
    index = [None for i in tweets]
    for i, tweet in enumerate(tweets):
        index[i] = lowesttoken(tweet, wordcount)
    others = collection()
    lowscores = set(index)
    T0 = time.time()
    for tweet in tweets:
        tweettokens = set(tweet.tokens)
        for token in tweettokens:
            if token in lowscores:
                others.add(token, tweettokens)
    for i, tweet in enumerate(tweets):
        tweettokens = set(tweet.tokens)
        for other in others[index[i]]:
            if len(other) > len(tweettokens) and tweettokens.issubset(other):
                break
        else:
            pruned.append(tweet)
    return pruned
                
def genTokens(dataset, ignore=set(["a", "an", "the", "and", "to", "is", "ll", "he", "his", "its", "im", "he", "she", "you", "them", "with", "but", "it", "i", "me", "that", "my", "not", "don", "we", "they", "didnt", "in", "be", "was", "been", "t", "am", "dont"])):
    for tweet in dataset:
        for token in tweet.tokens:
            if not token in ignore:
                yield token
        yield "###"

def showTweets(tokens, dataset):
    for tweet in dataset:
        for i in range(len(tweet.tokens)-len(tokens)):
            if tokens == tweet.tokens[i: i+len(tokens)]:
                print(tweet)
    
        


                
