from basics.utilities import *
import sklearn.svm
from sklearn.metrics.pairwise import cosine_similarity
from sklclassifiers import sklearnclassifier, tweets
from warnings import filterwarnings
filterwarnings("always") 
from scipy import sparse

class SVMCLASSIFIER(sklearnclassifier.SKLEARNCLASSIFIER):
    
    def __init__(self, train, params={"useDF":True}):
        self.readTrainingData(train, params=params)
        """
        Make an SVM
        """
        self.params = params
        max_iter = checkArg("max_iter", self.params, 20000)
        self.clsf = sklearn.svm.LinearSVC(max_iter=max_iter)
        self.clsf.predict_proba = self.clsf.decision_function
        """
        And get it to learn from the data
        """
        self.clsf.fit(self.matrix, self.values)
        self.weights = self.clsf.coef_

    def binaryScores(self, scores):
        return [1, 0] if scores < 0 else [0, 1]

def showSupport(self):
    decision_function = self.clsf.decision_function(self.matrix)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    support_vector_indices = numpy.where(numpy.abs(decision_function) <= 1 + 1e-15)[0]
    return self.matrix[support_vector_indices]
    
def svmtests(train, test, n=5000, step=1.5, end=500000):
    d = {}
    t = {}
    while n<end:
        T0 = time.time()
        d[int(n)] = SVMCLASSIFIER(train, N=int(n))
        t[int(n)] = test
        print(int(n), int(time.time()-T0));
        n= n*step
    return d, t


        

