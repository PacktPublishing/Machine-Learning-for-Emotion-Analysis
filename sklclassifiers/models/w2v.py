
from basics.utilities import *
from basics import corpora, readers
reload(readers)
from numpy import log, array, zeros
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import math

from numpy import dot
from numpy.linalg import norm

class SIMILARITYMEASURE:

    def nearest(self, word, N=5, args=None):
        if isinstance(word, str):
            try:
                return [x[0] for x in self.most_similar(word, N=N, args=args)]
            except:
                return (word, [])
        else:
            return [self.nearest(x, N=N, args=args) for x in word]

class W2VMODEL(SIMILARITYMEASURE):

    def __init__(self, corpus=os.path.join(corpora.CORPORA, "TEXTS", "BNC"), sentences=None, N=1000000):
        self.corpus = corpus
        T0 = time.time()
        if sentences is None:
            print("reading corpus")
            sentences = readers.reader(self.corpus, readers.BNCSentenceReader, showprogress=True, pattern=".*.xml")
            sentences = [sentence.strip().lower().split() for sentence in listN(sentences, N)]
            T1 = time.time()
            print("\ntook %s"%(elapsedTime(time.time()-T0)))
        self.sentences = sentences
        T0 = time.time()
        print("making model")
        self.model = gensim.models.Word2Vec(sentences).wv
        print("took %s"%(elapsedTime(time.time()-T0)))

    def nearest(self, word, N=5):
        return self.model.most_similar(word, topn=N)

class GLOVEMODEL(SIMILARITYMEASURE):
    
    def __init__(self):
        self.glove = {}
        for l in open(os.path.join(corpora.CORPORA, "DATA", "GLOVE", "glove.6B.100d.txt")):
            l = l.split()
            self.glove[l[0]] = array([float(n) for n in l[1:]])
            if len(self.glove) % 1000 == 0:
                sys.stdout.write("\r%s"%(len(self.glove)))

    def cos_sim(self, a, b):
        a = self.glove[a]
        b = self.glove[b]
        return dot(a, b)/(norm(a)*norm(b))

    def nearest(self, target, N=10):
        matches = {other: self.cos_sim(target, other) for other in self.glove}
        l = [(word, score) for (word, score) in sortTable(matches)[1:N+1]]
        return l[:N]

def table2sparse(table, n):
    data = []
    rows = []
    cols = []
    for c in sorted(table.keys()):
        data.append(table[c])
        rows.append(0)
        cols.append(c)
    return sparse.csr_matrix((data, (rows, cols)), shape=(1,n+1))

def allnearest(models, words):
    if not isinstance(models, list):
        models = [models]
    if not isinstance(words, list):
        words = [words]
    print("\t%s"%("\t".join(words)))
    for model in models:
        nlist = []
        for word in words:
            try:
                n = [x if isinstance(x, str) else x[0] for x in model.nearest(word, N=10, show=False)]
                while n[0].startswith(word) or n[0][0].isupper() or (n[0] == "ate" and word == "eat"):
                    n = n[1:]
                nlist.append(n[0])
            except:
                nlist.append("--")
        print("%s\t%s"%(model.__class__.__name__, "\t".join(nlist)))
        
    
