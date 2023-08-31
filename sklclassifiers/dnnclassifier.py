from basics.utilities import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.patches
from sklclassifiers import svmclassifier, tweets, svmdemo
from warnings import filterwarnings
filterwarnings("ignore") 

class DNNCLASSIFIER(svmclassifier.SVMCLASSIFIER):
    
    def __init__(self, train=None, params=None):
        params0 = {"N":sys.maxsize, "wthreshold":5,
                 "useDF": False, "max_iter":100,
                 "solver":"sgd", "alpha": 1e-5}
        for k in params0:
            if not k in params:
                params[k] = params0[k]
        iter = params["max_iter"]
        self.params = params
        self.readTrainingData(train, params)
        hiddenlayers = params["hiddenlayers"]
        if not isinstance(hiddenlayers, tuple):
            hiddenlayers = hiddenlayers(self)
        # making a multi-layer classifier requires a lof of parameters to be set"
        self.clsf = MLPClassifier(solver=params["solver"],
                                  alpha=params["alpha"],
                                  max_iter=iter,
                                  hidden_layer_sizes=hiddenlayers,
                                  random_state=1)
        if iter == 1:
            self.clsf.partial_fit(self.matrix, self.values, classes=[i for i in range(len(self.train.emotions))])
        else:
            # print("Doing full fit")
            self.clsf.fit(self.matrix, self.values)
        z = [numpy.zeros(len(self.train.reducedindex))]
        w = []
        for i in range(len(z[0])):
            z[0][i] = 1
            w.append(self.clsf.predict_proba(z)[0])
            z[0][i] = 0
        self.weights = numpy.array(w).transpose()

def showdnn(self, words0=None, sigfig=2):
        print("words0 %s"%(words0))
        f = "%%.%sf"%(sigfig)
        Xwidth = 8
        fontsize=7
        fig = plt.figure(figsize=(Xwidth, Xwidth/1.5))
        Y = 75
        X = 50
        ax = plt.axes()
        ax.set_aspect("equal",adjustable='box')
        plt.axis("off")
        plt.ylim((0, X))
        plt.xlim((0, Y))
        nodes = [[]]
        ystep = Y/(len(self.clsf.coefs_))
        r = self.train.reducedindex
        index = {r[x]:x for x in r}
        loweralpha = re.compile("^[a-z]*$")
        if words0 is None:
            words0 = [index[i] for i in range(self.clsf.n_features_in_) if loweralpha.match(index[i])]
            words0.sort()
            words0.reverse()
        if len(words0) > 10:
            words1 = words0[:2]+["...", "..."]+words0[-2:]
        else:
            words1 = words0
        for i, layer in enumerate(self.clsf.coefs_):
            nodes.append([])
            if i == 0:
                layer = list(words1)
            s = X/(len(layer))
            y = ystep*i+4
            for j, k in enumerate(layer):
                x = s/2+j*s
                k = layer[j]
                if i == 0:
                    ax.text(y, x, k, ha='right', va='center',)
                else:
                    ax.add_patch(matplotlib.patches.Circle((y, x), radius=0.5, fill=False))
                    ax.text(y-2, x-2, f%(self.clsf.intercepts_[i-1][j]), fontsize=fontsize)
                if not k == "...":
                    nodes[-1].append((y, x, i, j, k))
        s = X/(self.clsf.n_outputs_)
        i += 1
        y = ystep*i-2
        nodes.append([])
        for j in range(self.clsf.n_outputs_):
            x = s/2+j*s
            k = self.train.emotions[j]
            ax.text(y-1, x-3, (f%(self.clsf.intercepts_[-1][j])).ljust(5), fontsize=fontsize)
            ax.text(y, x, k, ha='left', va='center',)
            nodes[-1].append((y, x, i, j, k))
        links = []
        for i in range(1, len(nodes)):
            links.append([])
            if len(nodes[i-1]) > 0:
                for (x0, y0, i0, j0, k0) in nodes[i-1]:
                    x0 = x0+1
                    for j, (x1, y1, i1, j1, k1) in enumerate(nodes[i]):
                        x1 = x1-1
                        svmdemo.plotpoints(plt, [(x0, y0), (x1, y1)])
                        c = f%(self.clsf.coefs_[i0][j0,j1])
                        a = (y0-y1)/(x0-x1)
                        b = y0-a*x0
                        x = x0+(x1-x0)*0.2
                        y = a*x+b
                        t = plt.text(*(x, y), c, ha="center", va="center", fontsize=fontsize)
                        t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white'))
                        links[-1].append([[i0, j0, k0], [i1, j1, k1]])
        # self.showCoeffs(words1)
        plt.show()
        plt.savefig("plt.jpg")
        plt.close()

def showCoeffs(self, words):
        if isinstance(words, str):
            words = words.split()
        print("\t%s"%("\t".join("%+6s"%("%s"%(e,)) for e in self.train.emotions)))
        print("\t%s"%("\t".join("%+6s"%("%.3f"%(i)) for i in self.clsf.intercepts_[0])))
        for word in words:
            if not word == "...":
                try:
                    print("%s\t%s"%(word, "\t".join("%+6s"%("%.3f"%(c)) for c in self.clsf.coefs_[0][self.train.reducedindex[word]])))
                except:
                    print("%s\t%s"%(word, "\t".join("-" for i in range(len(self.train.emotions)))))
