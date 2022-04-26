"""
A collection of generally useful things. 
"""
import os, sys, re
import time
import random
import subprocess
from numpy import sqrt
from multiprocessing import Pool
PROCESSES = 4

"""
data is some sliceable entity, action is something you want to do to the slices, recombine is for putting the results back together: slicing numpy arrays is cheaper than slicing lists, so it may be worth considering converting data to a numpy array first
"""

def recombineLists(l):
    for x in l:
        for y in x:
            yield y

def recombineTables(l):
    r = {}
    for x in l:
        for k in x:
            r[k] = x[k]
    return r

def splitTask(data, action, N=PROCESSES, recombine=recombineLists):
    k = int(len(data)/N)+1
    data = [data[i:i+k] for i in range(0, len(data), k)]
    pool = Pool(N)
    results = recombine(pool.map(action, data))
    pool.terminate()
    return results


"""
Run an external program: all fairly standard stuff, but conveniently
packaged. Throws an exception if anything is sent down stderror.
"""
def execute(cmd):
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    x = p.communicate()
    if x[1] is "":
        return x[0]
    else:
        raise Exception(x[1])

"""
Useful when you've got a table like {"a": 1000, "the": 750, "cat":
200, ...} so you can see what the most common things are.

>>> sortTable({"a": 1000, "the": 750, "cat":200})
[(1000, 'a'), (750, 'the'), (200, 'cat')]
"""
def sortTable(table):
    l = [(k, v) for v, k in table.items()]
    return list(reversed(sorted(l)))

"""
Quite often need to normalise a table, though a range of different
factors can be useful. Set copy to False if you don't want it done in
place.
"""
def normaliseTable(table0, n, copy=False):
    if copy:
        table1 = {}
    else:
        table1 = table0
    for k in table0:
        table1[k] = table0[k]/n
    return table1

"""
Normalise so the maximum value is 1
"""
def setMax1(table, copy=False):
    return normaliseTable(table, max(table.values()), copy=copy)

"""
Normalise so the length is 1
"""
def unitlength(vector, copy=True):
    return normaliseTable(vector, sqrt(sum(x**2 for x in vector.values())), copy=copy)

"""
Normalise so they add up to 1
"""
def softmax(vector, copy=False):
    return normaliseTable(vector, sum(vector.values()), copy=copy)

"""
Get rid of low scoring elements of a table
"""
def prune(table0, N=0):
    table1 = {}
    for k, v in table0.items():
        if v >= N:
            table1[k] = v
    return table1

def printall(l):
    for x in l: print(x)

"""
stuff for turning a set of vectors into nicely plotted latex
(would probably be better to use psplot for things like this)
"""
def plotvectors(vectors):
    maxX = max(vector["X"] for vector in vectors)
    maxY = max(vector["Y"] for vector in vectors)
    print(r"""
\psaxes(%.2f, %.2f)
\rput[bl](0,0){\rnode{A}{}}"""%(1.1*maxX, 1.1*maxY))
    for i, vector in enumerate(vectors):
        x = vector["X"]
        y = vector["Y"]
        print(r"""
\rput[tr](%.2f, %.2f){\rnode{B%s}{(%.2f, %.2f)}}
\ncline[nodesepB=3pt]{->}{A}{B%s}"""%(x, y, i, x, y, i))

"""
To time for instance f(99) do

>>> timing(lambda: f(99), 1000)

You have to wrap it up in a lambda-expression to stop it just being
executed.
"""
def timing(g, n):
    T0 = time.time()
    for i in range(n):
        g()
    return (time.time()-T0)/n
  
