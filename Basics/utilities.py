"""
A collection of generally useful things. 
"""
import os, sys, re, shutil
import time
import random
import subprocess
from numpy import sqrt
import numpy
from multiprocessing import Pool
import importlib
from importlib import reload

PROCESSES = 4

"""
It is sometimes worth taking advantage of having multicores by
splitting a task into chunks and allocating them to separate
cores. Python lets you do this using the multiprocessing library. The
code below makes it easy to use this library where you have a SIMD
problem, where the "single instruction" is some Python function and
the "multiple data" is something that can be split into N chunks,
where N is the number of cores available to you. If you have a *very*
large amount of data then this may not work very well: using
generators can save space because you don't have to have all the data
in memory at one time, but it is tricky to split the data into chunks
without reading it all into memory, and likewise recombining the
results typically requires you to have the individual results in
memory. Allocating tasks to multicores works best if you have a small
amount of data that you want to do something complex to. Sometimes it
pays off, sometimes it doesn't: the only thing you can reasonably do
is try it and see.

data is some sliceable entity, action is something you want to do to
the slices, recombine is for putting the results back together:
slicing numpy arrays is cheaper than slicing lists, so it may be worth
considering converting data to a numpy array first. action must be a
standard function. In particular it can't be a lambda-expression, so
you have to be careful about how it is defined -- if there are other
arguments that are to be passed to each of the pooled tasks you should
pass them as a tuple in otherargs. See findMatches in lsa.py for an
example where doing this pays off.
"""

def recombineLists(l):
    for x in l:
        for y in x:
            yield y

def recombineTables(l):
    print("Recombining %s"%(len(l)))
    r = {}
    for x in l:
        for k in x:
            r[k] = x[k]
    return r
          
def splitTask(data, action, otherargs=(), N=PROCESSES, recombine=recombineLists):
    k = int(len(data)/N)+1
    return runTasks([(data[i:i+k],)+otherargs for i in range(0, len(data), k)], action, N=N, recombine=recombine)

def runTasks(args, action, N=PROCESSES, recombine=recombineLists):
    pool = Pool(N)
    results = recombine(pool.map(action, args))
    pool.terminate()
    return results

"""
Run an external program: all fairly standard stuff, but conveniently
packaged. Throws an exception if anything is sent down stderror.
"""
def execute(cmd, ignore=[]):
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    x = p.communicate()
    if len(x[1]) == 0:
        return x[0]
    else:
        if not x[1].decode() in ignore:
            raise Exception(x[1].decode())

class counter(dict):

    def add(self, x, n=1):
        if x in self:
            self[x] += n
        else:
            self[x] = n

class collection(dict):
    
    def add(self, x, y):
        if x in self:
            self[x].append(y)
        else:
            self[x] = [y]

class setcollection(dict):
    
    def add(self, x, y):
        if x in self:
            self[x].add(y)
        else:
            self[x] = set([y])

class confusion(dict):

    def add(self, x, y):
        if not x in self:
            self[x] = counter()
        self[x].add(y)
            
            
"""
Useful when you've got a table like {"a": 1000, "the": 750, "cat":
200, ...} so you can see what the most common things are.

>>> sortTable({"a": 1000, "the": 750, "cat":200})
[(1000, 'a'), (750, 'the'), (200, 'cat')]
"""
def sortTable1(table, rev=False):
    l = list(sorted([(k, v) for v, k in table.items()]))
    if rev:
        l.reverse()
    return list([(v, k) for (k, v) in l])

def sortTable(t):
    l = [(t[x], x) for x in t]
    l.sort()
    l.reverse()
    l = [(x[1], x[0]) for x in l]
    return l

def sortTableToTable(t, N=10):
    return {x[0]: x[1] for x in sortTable(t)[:N]}

def printST(st, format="%.2f"):
    return ", ".join(("%s:"+format)%(x[0], x[1]) for x in st)

def delete(x, l):
    return [y for y in l if not y == x]

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

def normalise(table, copy=True):
    return normaliseTable(table, sum(table.values()), copy=copy)

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

def printTable(d, n=sys.maxsize):
    for x in list(d.keys())[:n]:
        print(x, d[x])

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
def timing(g, n=1):
    T0 = time.time()
    for i in range(n):
        g()
    return (time.time()-T0)/n

def elapsedTime(secs):
    if secs > 60:
        return "%s mins %s secs"%(secs//60, int(secs%60))
    elif secs > 1:
        return "%.1f secs"%(secs)
    else:
        return "%.3f secs"%(secs)

def progress(i, N, T0, step=1000):
    if N and i > 0 and i % step == 0:
        secs = (N-i)*(time.time()-T0)/i
        sys.stdout.write("\r%.2f done: %s remaining"%(i/N, elapsedTime(secs)))
  
class UniversalSet(set):
    def __and__(self, other):
        return other

    def __rand__(self, other):
        return other

    def __contains__(self, x):
        return True

def identity(x):
    return x
        
def safelog(threshold):
    return 0 if threshold == 0 else numpy.log(threshold)

def deindex(x, l):
    for i, y in enumerate(l):
        if x == y:
            return i
    else:
        raise Exception("%s not in %s"%(x, l))

def checkArg(arg, args, default=False):
    try:
        return args[arg]
    except:
        return default

def makedirs(d, showIfExists=False):
    try:
        os.makedirs(d)
    except:
        if showIfExists:
            print("%d already exists")
