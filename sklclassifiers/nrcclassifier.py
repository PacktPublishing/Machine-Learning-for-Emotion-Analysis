from basics.utilities import *
from sklclassifiers import baseclassifiers, metrics, tweets
from basics import a2bw
from chapter4 import arabicstemmer, spanishstemmer, stem3, tokenisers
from basics.corpora import CORPORA

"""
EMOLEX comes in various forms. We are using the one where the first column
is an English word, the next 11 are the values for each emotion and the last
is a translation of the given English word into some other language. The
default is the one where the other language is Arabic, but we have done some
experiments with a Spanish corpus, for which we need a Spanish stemmer. The way to
extend this to other languages should be obvious.

params is a set of parameters for applying the algorithm in different s
"""
EMOLEX=os.path.join(CORPORA, "DATA/NRC-Emotion-Lexicon/OneFilePerLanguage/%s")
def readNRC(ifile=EMOLEX%("Arabic-NRC-EmoLex.txt"), targets=None, params=False):
    lines = list(open(ifile))
    """
    emotions is the list of emotions in the first line of the file
    targets is the list of emotions in the dataset that the classifier
    is going to be applied to. They have to be the same, so we reduce them
    to the set that appears in both.
    """
    emotions = lines[0].strip().split("\t")[1:-1]
    if isinstance(targets, str):
        targets = targets.split()
    emotionIndex = [True if e in targets else False for e in emotions]
    targetIndex = [True if e in emotions else False for e in targets]
    lex = {}
    """
    add entries line by line
    """
    for line in lines[1:]:
        line = line.split("\t")
        text =line[0]
        if params["language"] == "AR":
            text = a2bw.convert(text, a2bw.a2bwtable)
            if params["stemmer"] == "standard":
                text = arabicstemmer.stemAll(text, arabicstemmer.TWEETGROUPS)
        elif params["language"] == "ES":
            if params["stemmer"] == "standard":
                text = spanishstemmer.stemAll(text)
        else:
            if params["stemmer"].startswith("standard"):
                # Don't actually want to strip off prefixes for sentiment classification
                stem3.PREFIXES = {}
                text = stem3.stemAll(text, stem3)
            elif params["stemmer"].startswith("morphy"):
                text = stem3.morphyroot(text)
        lex[text] = [int(x) for (x, y) in zip(line[1:-1], emotionIndex) if y]
    return lex, emotionIndex, targetIndex

class NRCCLASSIFIER(baseclassifiers.BASECLASSIFIER):

    def __init__(self, train, ifile=EMOLEX%("Arabic-NRC-EmoLex.txt"), params=None):
        self.threshold = params["threshold"]
        if isinstance(train, str):
            train = tweets.makeDATASET(train, params=params)
        self.train = train
        self.params = self.train.params
        if params["language"] == "ES":
            ifile=EMOLEX%("Spanish-NRC-EmoLex.txt")
        self.lex, self.emotionIndex, self.targetIndex = readNRC(targets=self.train.emotions, ifile=ifile, params=params)

    def applyToTweet(self, tweet, threshold=0.5, probs=True):
        score = [0]*len(self.train.emotions)
        for token in tweet.tokens:
            try:
                for i, x in enumerate(self.lex[token]):
                    score[i] += x
            except:
                pass
        best = max(score)
        if best > 0:
            score = [1 if x >= best*threshold else 0 for x in score]
        return score
