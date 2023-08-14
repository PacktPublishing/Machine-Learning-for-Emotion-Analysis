import re, os, sys
from nltk.corpus import wordnet
from corpora import *

"""
Make a copy rather than doing it in place because reading the corpus
is time consuming and we don't want to mess it up when we do the
pruning
"""
def prunewords(words0, n=100):
    words1 = {}
    for w in words0:
        if words0[w] >= n:
            words1[w] = words0[w]
    return words1

def addPair(prev, word, words, pairs, n=sys.maxsize):
    if prev in words and word in words:
        pair = "%s-%s"%(prev, word)
        pairs.add(pair)
        if len(pairs)%n == 0:
            print("%s found"%(len(pairs)))
"""
Collect and count bigrams
"""
def bigrams(wordlist, words=None, pairs=None, threshold=100):
    if pairs is None:
        pairs = counter()
    if words is None:
        words = allwords(wordlist)
    words = prunewords(words, n=threshold)
    prev = False
    az = re.compile(".[a-z؟-ۼ]*$")
    for word in wordlist:
        if prev and az.match(prev) and az.match(word):
            addPair(prev, word, words, pairs)
        prev = word
    return prunewords(pairs, n=threshold)
        
"""
Calculate the probabily of x happening. t is how often it happened, tsum is total number of instances
"""
def prob(x, t, tsum):
    return float(t[x])/tsum

from math import log

"""
pair is "potted-meat", words is all the words there are. We pass in the sizes of words and pairs
because calculating it each time is time-consuming. 

We assume that pairs are stored as a string with a - in, e.g. "potted-meat". In a few cases one of the
constituent words itself contains a dash, in which case we won't be able to split it into two just
by using split. We just skip over those cases. Doesn't make any difference, they're not very common or
very significant.
"""
def pmiPair(pair, words, wsum, pairs, psum):
    try:
        w1, w2 = pair.split("-")
    except:
        return -1
    return log(prob(pair, pairs, psum)/(prob(w1, words, wsum)*prob(w2, words, wsum)))
        
def allpmi(pairs, words):
    wsum = sum(words.values())
    psum = sum(pairs.values())
    l = [(pmiPair(pair, words, wsum, pairs, psum), pair, pairs[pair]) for pair in pairs if pair.lower()==pair]
    return [x for x in reversed(sorted(l)) if x[0] > 0]
        
def getlexicon(words):
    l = counter()
    for word in words:
        if isinstance(word, tuple):
            word = word[0]
        l.add(word)
    return l

def thresholdpmi(pmi, n):
    return [p for p in pmi if p[-1] >= n]

def doItAllPMI(wordlist, t1=50, t2=100):
    print("%s words"%(len(wordlist)))
    words = getlexicon(wordlist)
    print("get rid of the top %s words because they will make too many pairs (e.g. 'of-the')"%(t2))
    topscorers = set([x[0] for x in sortTable(words)[:t2]])
    print("TOP SCORERS: %s"%(topscorers))
    wordlist = [word for word in wordlist if word in ["S-START", "S-END"] or not word in topscorers]
    print("%s distinct words found (%s tokens)"%(len(words), sum(words.values())))
    print("Getting pairs that occur at least %s times"%(t1))
    pairs = bigrams(wordlist, words=words, threshold=t1)
    print("%s pairs found"%(len(pairs)))
    print("Calculating PMI")
    pmi = allpmi(pairs, words)
    pmiTable = {p[1]: (p[0], p[2]) for p in pmi}
    return pmi, pmiTable, words, pairs

sample = "新冠疫情｜如心酒店集團向染疫房客收5000元消毒費　旗下非檢疫酒店適用"

punct = set(["[", ".", ","])
def prefix2string(prefix):
    while set(prefix).intersection(punct):
        prefix = prefix[1:]
    return " ".join(prefix)

def suffix2string(suffix):
    while set(suffix).intersection(punct):
        suffix = suffix[:-1]
    return " ".join(suffix)
    
def compounds(wordreader, keys=False):
    print("reading words")
    allwords = list(wordreader)
    words = set()
    pairs = set()
    collocations = collection()
    chars = re.compile("[a-z]*")
    last = False
    print("making tables")
    for i, word in enumerate(allwords):
        if len(word) > 4 and chars.fullmatch(word):
            if last:
                pair = "%s %s"%(last, word)
                pairs.add(pair)
                collocations.add(pair, (prefix2string(allwords[i-6:i-1]), pair, suffix2string(allwords[i+1:i+7])))
            collocations.add(word, (prefix2string(allwords[i-6:i]), word, suffix2string(allwords[i+1:i+7])))
            words.add(word)
        last = word
    print(len(words), len(pairs), len(collocations))
    compounds = set()
    for word in words:
        for i in range(3, len(word)-2):
            w1, w2 = word[:i], word[i:]
            if w1 in words and w2 in words:
                pair = "%s %s"%(w1, w2)
                if pair in pairs:
                    compounds.add("%s: %s-%s"%(pair, w1, w2))
    return sorted(compounds), sorted(pairs), collocations

def showCollocations(words, collocations):
    examples = []
    for word in words:
        w0, w1 = word.split(" ")
        if len(collocations[word]) > 3 and len(collocations["%s%s"%(w0, w1)]) > 3:
            examples += collocations[word][:4]
            examples += collocations["%s%s"%(w0, w1)][:4]
    mx = my = 0
    for x, y, z in examples:
        mx = max(mx, len(x))
        my = max(my, len(y))
    for x, y, z in examples:
        x = " "*(mx-len(x))+x
        print ("%s %s %s"%(x, y, z))

    
    
