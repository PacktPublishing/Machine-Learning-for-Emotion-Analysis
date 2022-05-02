"""
regexparser will give us tables of VERB:OBJECT or OBJECT:VERB or
... pairs. We want to use those for calculating similarities, either
directly or after doing LSA (i.e. after using SVD do get a diagonal
matrix, deleting the lower parts and then reconstructing the original
(i.e. GlOVe))

This all only works if you have a lot of data. And if you have a lot of data then processing it will take a lot of memory and a lot of time. So there's quite a lot of stuff in here to make things run as fast as possible as use as little memory as possible. In particular, we have to manipulate quite large matrices (e.g. 50K*50K = 2500000000 entries), which take a long time to iterate through and take up a lot of room. 

We start by reevaluating the scores by TF-IDF.
"""

from utilities import *
from corpora import *
from numpy import log, array, zeros
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from regexparser import *
from taggers import TAGGER
import gc

"""
Just count how many documents (pairs) contain each word. Default
involves taking logs, but there is no very convincing reason why this
should be used, so you can do other things (e.g. just keep the raw
score) if you want.
"""
def getIDF(pairs, uselog=log):
    idf = {}
    for document in pairs.values():
        for x in set(document):
            if x in idf:
                idf[x] += 1
            else:
                idf[x] = 1
    N = float(len(pairs))
    for w in idf:
        idf[w] = uselog(N/idf[w])
    return idf

"""
default is to make a copy rather than doing it in situ: regexParse
returned tables, one per group (e.g. one for TVERB, one for OBJ,
...). applyIDF deals with *one* such table, applyAllIDF does the lot.
"""
def applyIDF(pairs0, copying=True, uselog=log):
    idf = getIDF(pairs0, uselog=uselog)
    if copying:
        pairs1 = {}
    else:
        pairs1 = pairs0
    for w in pairs0:
        pairs1[w] = {}
        for x in pairs0[w]:
            pairs1[w][x] = pairs0[w][x]*idf[x]
    return pairs1

def applyAllIDF(pairs0, copying=True, uselog=log):
    if copying:
        pairs1 = {}
    else:
        pairs1 = pairs0
    for g in pairs0:
        pairs1[g] = applyIDF(pairs0[g], uselog=uselog)
    return pairs1

"""
Our coccurrence tables link words to words. For doing the sums we need
matrices (and sparse matrices). So we map words to indexes so we can
then convert cooccurrence tables to matrix: dimensions maps words to
indices, invdimensions maps indices to words.

>>> pairs = {"a": {"x": 2, "y":1}, "b": {"x": 3, "z": 4}}
>>> pairs
{'a': {'x': 2, 'y': 1}, 'b': {'x': 3, 'z': 4}}
>>> dimensions, invdimensions = getDimensions(pairs)
>>> dimensions
{'a': 0, 'x': 1, 'y': 2, 'b': 3, 'z': 4}
>>> invdimensions
{0: 'a', 1: 'x', 2: 'y', 3: 'b', 4: 'z'}
"""

def getDimensions(pairs):
    dimensions = {}
    """ 
    pairs is a table of word: set of cooccurrences pairs, where a
    set of coocurrence pairs is a table of word: count pairs

    dimensions is to be a mapping from words to integers which we will
    use when converting a table VERB:OBJECT or OBJECT:VERB pairs to a
    matrix, invdimensions is just the inverse.
    """
    for key in pairs:
        if not key in dimensions:
            dimensions[key] = len(dimensions)
        for word in pairs[key]:
            if not word in dimensions:
                dimensions[word] = len(dimensions)
    invdimensions = {dimensions[k]: k for k in dimensions}
    return dimensions, invdimensions

"""
Convert a table of HD:DTR pairs to a SPARSE matrix. These table
contain large numbers of zero entries, so it would be madness not to
store them as sparse matrices. Keep hold of the mappings between words
and indices.
"""
def pairs2matrix(pairs):
    dimensions, invdimensions = getDimensions(pairs)
    data = []
    rows = []
    cols = []
    for word in pairs:
        row = dimensions[word]
        for other in pairs[word]:
            data.append(pairs[word][other])
            cols.append(dimensions[other])
            rows.append(row)
        data.append(0)
        cols.append(len(dimensions))
        rows.append(row)
    return dimensions, invdimensions, sparse.csc_matrix((data, (rows, cols)))

"""
Do it to several sets of pairs: we store the VERB:OBJ and OBJ:VERB
pairs (or VERB:SUBJ and SUBJ:VERB, or ... pairs) together, so we
should do the same things to both while we're about it.
"""
def allpairs2matrices(grouped):
    dimensions = {}
    invdimensions = {}
    matrices = {}
    for g in grouped:
        d, i, m = pairs2matrix(grouped[g])
        dimensions[g] = d
        invdimensions[g] = i
        matrices[g] = m
    return dimensions, invdimensions, matrices

"""
Calculate the similarity between two words: use dimensions to get from
the words into the sparse matrix of cooccurrence vectors, use the
standard cosine_similarity.
"""
def score(word, other, dimensions, matrix):
    return cosine_similarity(matrix[dimensions[word]],
                             matrix[dimensions[other]])

def showSortedTable(l):
    return ", ".join("%s %.2f"%(y, x) for (x, y) in l)

"""
matches = findMatches("king", "TVERB", PAIRS1, DIMENSIONS, MATRICES, show=True)

Using splitTask to split the list of possible matches into chunks and
run searchPairsForMatches on the chunks can speed things up -- a bit
over twofold improvement on a four core Mac (you don't tend to get an
N-fold speedup with N cores).
"""

def searchPairsForMatches(data):
    """ 
    data is a bundle or arguments because splitTask wants all the
    arguments supplied as a single collection
    """
    (others, word, group, pairs, dimensions, matrix) = data
    matches = {}
    """
    make a table of words: (score, other word, number of terms in the cooccurrence set for other)

    We keep the count of the terms in the coccurrences just for
    inspection
    """
    for i, other in enumerate(others):
        if not other == word:
            matches[other] = (score(word, other, dimensions[group], matrix[group])[0][0], other, len(pairs[group][other]))
    return matches

"""
split the set of possible matches and run searchPairsForMatches in
each set. It's worth doing because while we do have to work hard we
don't have to make copies of large data structures.
"""
def splitTask(data, action, otherargs=(), N=PROCESSES, recombine=recombineLists):
    k = int(len(data)/N)+1
    data = [(data[i:i+k],)+otherargs for i in range(0, len(data), k)]
    pool = Pool(N)
    results = pool.map(action, data)
    results = recombine(results)
    pool.terminate()
    return results

"""
Find matches for the target word, either as a single process or by
splitting it into chunks, and print out the top few cases.

>>> m = findMatches("king", "TVERB", PAIRS2, DIMENSIONS, MATRICES)
Best matches for king (become 234.44, crown 234.22, save 82.40, say 71.55, marry 62.62)
(0) queen: 0.68 (277, save 116.95, crown 90.50, become 66.35, award 55.03, won 52.85)
(1) emperor: 0.64 (152, crown 69.20, elect 35.30, depose 26.54, become 20.64, please 17.24)
(2) member: 0.57 (713, become 718.07, elect 370.62, recruit 213.11, ask 177.65, comprise 165.88)
(3) kadi: 0.55 (6, become 28.02, appoint 6.48, say 5.30, fought 4.40, show 1.64)
(4) millionaire: 0.54 (25, become 53.08, marry 22.10, mad 15.58, born 10.37, die 8.66)
(5) fundholders: 0.54 (6, become 20.64, expect 2.23, develop 2.16, help 2.16, allow 1.64)
(6) commonplace: 0.54 (5, become 39.81, assault 4.86, mix 3.81, apply 3.26, mad 1.56)
(7) champion: 0.53 (151, defend 146.71, crown 106.47, become 106.16, beat 69.71, race 38.78)
(8) monk: 0.53 (82, become 50.13, reply 19.01, ask 17.04, say 14.58, heard 10.78)
(9) patron: 0.52 (61, become 50.13, seek 10.16, roxburghe 9.68, welcome 9.07, have 7.47)
(10) anorexic: 0.52 (8, become 32.44, fail 8.03, recover 7.37, term 3.91, resemble 3.51)
(11) archbishop: 0.51 (74, become 56.03, appoint 29.15, consult 14.93, elect 13.24, request 10.79)
(12) partner: 0.51 (297, become 154.82, seek 73.67, find 65.38, have 47.67, need 30.31)

The top few entries in the cooccurrence table for "king" are (become 234.44, crown 234.22, save 82.40, say 71.55, marry 62.62), the cooccurrence table for "queen" contains 277 elements of which the top few are save 116.95, crown 90.50, become 66.35, award 55.03, won 52.85, "queen" is the word that is most similar to "king".

Using the basic cooccurrence table gives reasonable looking nearest
neighbours, but there are things that look a bit odd:

>>> m = findMatches("queen", "TVERB", PAIRS2, DIMENSIONS, MATRICES)
Best matches for queen (save 116.95, crown 90.50, become 66.35, award 55.03, won 52.85)
(0) king: 0.68 (497, become 234.44, crown 234.22, save 82.40, say 71.55, marry 62.62)
(1) master: 0.46 (322, become 89.94, serve 57.45, appoint 51.82, won 46.97, please 38.78)
(2) rainforest: 0.45 (29, save 45.19, destroy 16.69, harm 8.96, protect 7.97, raze 7.28)
(3) emperor: 0.45 (152, crown 69.20, elect 35.30, depose 26.54, become 20.64, please 17.24)
(4) planet: 0.43 (133, save 85.06, inhabit 23.28, destroy 22.25, plunder 15.68, paralysing 14.97)

"king", "master", "emperor", "lady" are reasonable candidates for "queen", but "rainforest"? "planet"?
"""
def findMatches(word, group, pairs, dimensions, matrix, show=True, split=True, topN=5):
    # Search through all the possible matches
    if split:
        # Split the possible matches into N chunks and run them on N cores
        # (best results come from setting N to the actual number of physical cores)
        matches = splitTask(array(list(pairs[group].keys())), searchPairsForMatches, (word, group, pairs, dimensions, matrix), recombine=recombineTables, N=4)
    else:
        # Just enumerate all the way the set of pairs as a single process
        matches = searchPairsForMatches((pairs[group], word, group, pairs, dimensions, matrix))
    if show:
        pruned = [(s, f, l) for ((s, f, l), x) in sortTable(matches)]
        print("Best matches for %s (%s)"%(word, showSortedTable(sortTable(pairs[group][word])[:5])))
        for i, (s, f, l) in enumerate(pruned[:topN]):
            print("(%s) %s: %.2f (%s, %s)"%(i, f, s, l, showSortedTable(sortTable(pairs[group][f])[:5])))
    return matches

"""
Remove words which have at most threshold entries in their
cooccurrence tables. Rationale is that words which have very sparse
cooccurrence tables are likely to be unreliable, but the real reason
is that doing LSA involves manipulating *large* matrices, and remove
cases with very small cooccurrence tables saves a lot of space without
losing very relevant items.
"""
def pruneTable(table0, threshold=3):
    table1 = {}
    for k in table0:
        if len(table0[k]) > threshold:
            table1[k] = table0[k]
    return table1

def pruneTables(tables):
    return {g: pruneTable(tables[g]) for g in tables}

"""
STUFF FOR ACTUALLY DOING LSA
"""

"""
The easy bit: get the SVD of cooccurrence matrix. We just use the top
N elements of the diagonal.
"""
def getsvd(matrix, N=100):
    u, d, vh = svds(matrix, N)
    return u, d, vh

"""
The matrix that we get by reconstructing using U, the significant
parts of D and VH tends to have a lot of entries that are
insignificant but aren't actually zero. It's worth replacing them by
actual zeros before we convert to a sparse matrix. This is quite time
consuming (so we printt out a progress bar) but is well worth doing,
since otherwise the matrix is not very sparse do doing similarity
measures is *very* time-consuming.
"""
def nearzeros(A0, threshold=10):
    data = []
    rows = []
    cols = []
    print("collecting non-zeros from %s"%(A0.shape[0]))
    N = int(A0.shape[0]/100)
    for r, row in enumerate(A0):
        if r % N == 0:
            sys.stdout.write("\r%.2f"%(float(r)/A0.shape[0]))
        for c, d in enumerate(row):
            if d > threshold:
                data.append(d)
                rows.append(r)
                cols.append(c)
    print("constructing sparse version")
    return sparse.csc_matrix((data, (rows, cols)), A0.shape)

"""
The scripts for doing the whole thing are quite slow, and can easily
fall over. We therefore return the results as the values of global
variables. This is in general a bad idea, but it's handy here because
it means that if some steps work and then the whole thing falls over
at least the bits that have been done successfully are still
available.

The matrices that we get by dropping the less significant elements
from the diagonal from the SVD, recreating the cooccurrence matrix
using the u, d' and vh and then zeroing out the less significant parts
of the recreated cooccurrence table gives much better looking nearest
neighbours:

>>> m = findMatches("queen", "TVERB", PAIRS2, DIMENSIONS, MATRICES)
Best matches for queen (save 116.95, crown 90.50, become 66.35, award 55.03, won 52.85)
(0) king: 0.68 (497, become 234.44, crown 234.22, save 82.40, say 71.55, marry 62.62)
(1) master: 0.46 (322, become 89.94, serve 57.45, appoint 51.82, won 46.97, please 38.78)
(2) rainforest: 0.45 (29, save 45.19, destroy 16.69, harm 8.96, protect 7.97, raze 7.28)
(3) emperor: 0.45 (152, crown 69.20, elect 35.30, depose 26.54, become 20.64, please 17.24)
(4) planet: 0.43 (133, save 85.06, inhabit 23.28, destroy 22.25, plunder 15.68, paralysing 14.97)

>>> m = findMatches("queen", "TVERB", PAIRS2, DIMENSIONS, REDUCEDMATRICES)
Best matches for queen (save 116.95, crown 90.50, become 66.35, award 55.03, won 52.85)
(0) king: 0.75 (497, become 234.44, crown 234.22, save 82.40, say 71.55, marry 62.62)
(1) master: 0.68 (322, become 89.94, serve 57.45, appoint 51.82, won 46.97, please 38.78)
(2) mp: 0.63 (194, told 152.46, lobby 145.22, elect 132.36, become 88.47, say 39.75)
(3) judge: 0.62 (238, told 85.27, appoint 68.02, say 39.75, impress 34.47, ask 34.07)
(4) companion: 0.61 (142, travel 33.87, become 32.44, told 20.67, say 15.90, have 14.93)

>>> m = findMatches("eat", "OBJ", PAIRS2, DIMENSIONS, MATRICES, topN=7)
Best matches for eat (food 611.06, meal 538.59, breakfast 402.72, meat 366.89, lunch 327.96)
(0) cook: 0.68 (225, meal 286.62, dinner 167.94, food 86.64, breakfast 85.19, onion 83.88)
(1) prepare: 0.38 (959, meal 318.11, report 271.39, ground 221.63, plan 199.33, food 148.20)
(2) overcook: 0.35 (7, cabbage 4.87, pasta 4.76, food 4.56, vegetable 3.86, thing 3.69)
(3) materialize: 0.30 (4, food 4.56, battle 3.23, total 2.92, expression 2.80)
(4) munch: 0.29 (53, way 29.66, sandwich 23.77, munch 21.74, peanut 15.60, matzo 15.07)
(5) ladle: 0.28 (4, haricot 7.94, food 4.56, sympathy 3.73, meat 3.22)
(6) microwave: 0.27 (4, pud 4.56, steak 4.37, fish 2.39, food 2.28)

>>> m = findMatches("eat", "OBJ", PAIRS2, DIMENSIONS, REDUCEDMATRICES, topN=7)
Best matches for eat (food 611.06, meal 538.59, breakfast 402.72, meat 366.89, lunch 327.96)
(0) cook: 0.85 (225, meal 286.62, dinner 167.94, food 86.64, breakfast 85.19, onion 83.88)
(1) drink: 0.77 (254, tea 349.06, coffee 317.92, win 186.77, water 169.78, alcohol 160.19)
(2) drank: 0.71 (121, coffee 226.61, tea 221.52, win 91.90, cup 71.82, water 61.37)
(3) sip: 0.70 (87, coffee 301.01, tea 241.66, drink 202.22, win 118.58, champagne 110.18)
(4) pour: 0.69 (261, tea 238.30, water 233.19, scorn 204.87, coffee 186.02, win 127.48)
(5) prepare: 0.61 (959, meal 318.11, report 271.39, ground 221.63, plan 199.33, food 148.20)
(6) order: 0.55 (1084, retrial 130.97, arrest 125.06, investigation 102.60, inquiry 84.95, release 83.84)

The version with the reduced cooccurrence table links "eat" to various
words about drinking, i.e. to consumption of food and drink, where the
original links it to ways of preparing food.

This all works OK for words that occur a lot in the training data, but
even with 100 million words most words only occur a comparatively
small number of times, and if that happens then you can't count on
getting good matches. Add to that the fact that for sentiment analysis
we need to get the emotional overtones, whereas nearest matches tend
to be about the underlying type of a word, and maybe using this
approach to find similar words will not be as helpful as one might
wish for.
"""

def doLSA(matrices):
    global m0, u, d, vh, d1, m2, REDUCEDMATRICES
    REDUCEDMATRICES = {}
    for group in matrices:
        gc.collect()
        m0 = matrices[group]
        u, d, vh = getsvd(m0)
        """
        zero out the bottom bit of the diagonal
        """
        d1 = zeros(d.shape)
        for i in range(30, 100):
            d1[i] = d[i]
        s = (u * d1) @ vh
        u = d1 = vh = None
        gc.collect()
        REDUCEDMATRICES[group] = nearzeros(s)
        s = None

def doItAll(uselog=log):
    global TAGGED, tagger, PAIRS0, PAIRS1, PAIRS2, DIMENSIONS, INVDIMENSIONS, MATRICES, REDUCEDMATRICES
    T0 = time.time()
    print("MAKING TAGGER")
    tagger = TAGGER(reader(os.path.join(BNC), lambda w: BNCTaggedWordReader(w, specials={"to":"TO", "that": "THAT"})))
    T1 = time.time()
    print("MADE TAGGER -- TOOK %s SECONDS"%(int(T1-T0)))
    print("SETTING UP SENTENCE READER")
    SENTENCES = reader(os.path.join(BNC), BNCSentenceReader)
    print("SETTING UP TAGGER FOR SENTENCE READER")
    TAGGED = (tagger(sentence) for sentence in SENTENCES)
    print("READING, TAGGING AND PARSING")
    PAIRS0 = regexParse(OBJRULES, TAGGED)
    print(", ".join("%s: %s"%(g, len(PAIRS0[g])) for g in PAIRS0))
    PAIRS1 = pruneTables(PAIRS0)
    print(", ".join("%s: %s"%(g, len(PAIRS1[g])) for g in PAIRS1))
    T2 = time.time()
    print("TAGGED AND PARSED -- TOOK %s SECONDS"%(int(T2-T1)))
    PAIRS2 = applyAllIDF(PAIRS1, uselog=uselog)
    print("CONVERT TO SPARSE MATRICES")
    DIMENSIONS, INVDIMENSIONS, MATRICES = allpairs2matrices(PAIRS2)
    REDUCEDMATRICES = doLSA(MATRICES)

def useglove():
    glove = {}
    for l in open("DOWNLOADS/glove.6B.100d.txt"):
        l = l.split()
        glove[l[0]] = array([float(n) for n in l[1:]])
        if len(glove) % 1000 == 0:
            sys.stdout.write("\r%s"%(len(glove)))
    return glove

from numpy import dot
from numpy.linalg import norm
def cos_sim(a, b, glove):
    a = glove[a]
    b = glove[b]
    return dot(a, b)/(norm(a)*norm(b))

def nearestGloveMatches(target, glove):
    matches = {other: cos_sim(target, other, glove) for other in glove}
    return sortTable(matches)[:10]
