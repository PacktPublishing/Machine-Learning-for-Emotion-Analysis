import sys

"""
Dynamic time warping: mixture of code for doing it and stuff for
generating pretty LaTeX output from it.
"""

"""
You can define different ways of calculating the cost of the core
edit operations--EXCHANGE, INSERT and DELETE are functions that you apply
to the two otems beings swapped. Default definitions just return constants.
"""
EXCHANGE = (lambda x, y: 3)
INSERT = (lambda x: 2.0)
DELETE = (lambda x: 2.0)

def DELETE(x):
    return 2.0
    
"""
Two alternatives that you could use for EXCHANGE: the first one
utilises the fact that in MT strings of similar lengths are more
likely to be mutual translations than ones of different lengths, the
second uses DTW to match the strings themselves, on the grounds that
lots of words are either borrowed or derived from the same other
language, so that mutual translations are actually likely to look
pretty similar.
"""
def matchStrings(s0, s1):
    lens0 = float(len(s0))
    lens1 = float(len(s1))
    if lens0 == lens1:
        return 1
    elif lens0 > lens1:
        if lens1 == 0 or lens1 == 0.0:
            return 1
        return lens0/lens1
    else:
        if lens0 == 0 or lens0 == 0.0:
            return 1
        return lens1/lens0

def matchStrings1(s0, s1):
    a = ARRAY(s0, s1)
    a.findPath()
    return (a.last().value)/max(float(len(s0)), float(len(s1)))

def matchnums(n1, n2):
    try:
        n1 = float(n1)
    except:
        n1 = 100000
    try:
        n2 = float(n2)
    except:
        n2 = 100000
    return float(abs(n1-n2))/max(n1, n2)

def processShowAlignment(a1, a2, EXCHANGE=EXCHANGE):
    a = array(a1, a2, EXCHANGE=EXCHANGE)
    return a.showAlignment()

     
"""
One to use if you're matching arrays of numbers. Two numbers are similar
if their difference is small!
"""
def matchFloats(s0, s1, d=4.0):
    return abs(s0-s1)/float(d)

class LINK:

    """
    A LINK connects the points at array[x][y], and may contain
    a back-pointer and a score
    """
    def __init__(self, x, y, array):
        self.x = x
        self.y = y
        self.link = False
        self.value = -1
        self.cost = 0
        self.edit = 'EXCHANGE'
        self.array = array

    """
    Look to the right, diagonally and down and see if this is better
    predecessor of the place you're looking at than the one you've already
    got (if you have indeed already been there). This is the key operation
    in the entire algorithm
    """
    def extend(self, dx, dy):
        a = self.array
        array = a.array
        x = self.x+dx
        y = self.y+dy
        if x >= len(a.v1) or y >= len(a.v2):
            return
        av1x = a.v1[x]
        av2y = a.v2[y]
        if dx == 1 and dy == 1:
            edit = 'EXCHANGE'
            if av1x == av2y:
                s = 0
            else:
                s = a.EXCHANGE(av1x, av2y)
        elif dx == 0:
            edit = 'INSERT'
            s = a.INSERT(av1x)

        elif dy == 0:
            edit = 'DELETE'
            s = a.DELETE(av2y)

        else:
            raise Exception("No operator defined for %s, %s"%(dx, dy))

        other = array[x][y]
        if other.value == -1 or self.value+s < other.value:
            other.cost = s
            other.value = self.value+s
            other.link = self
            other.edit = edit

    """
    For tracing back from the end once you've got there
    """
    def getLinks(self):
        if self.link:
            links = self.link.getLinks()
            return links+[(self.x, self.y, self.cost, self.edit)]
        else:
            return [(self.x, self.y, self.cost, self.edit)]


class ARRAY:

    """
    An array of links. Set the scoring functions you're going to use
    for this problem.
    """
    def __init__(self, v1, v2, EXCHANGE=EXCHANGE, INSERT=INSERT, DELETE=DELETE):
        if v1.__class__.__name__ == "str":
            v1 = "#"+v1+"#"
            v2 = "#"+v2+"#"
        else:
            v1 = ["start"]+v1+["end"]
            v2 = ["start"]+v2+["end"]
        self.v1 = v1
        self.v2 = v2
        self.EXCHANGE=EXCHANGE
        self.INSERT=INSERT
        self.DELETE=DELETE
        self.array = [[LINK(i, j, self) for j in range(0, len(self.v2))] for i in range(0, len(self.v1))]

    """
    Walk through the links, seeing where you can get to and whether this
    is the best way to get there
    """
    def findPath(self):
        self.array[0][0].value = 0
        for j in range(0, len(self.array[0])-1):
            for i in range(0, len(self.array)):
                l = self.array[i][j]
                l.extend(0, 1)
                l.extend(1, 1)
                l.extend(1, 0)
        return self.array[-1][-1].value

    def last(self):
        row = self.array[len(self.array)-1]
        return row[len(row)-1]

    """
    Plain text output
    """
    def show(self):
        s = ''
        for j in range(0, len(self.array[0])):
            for i in range(0, len(self.array)):
                s = s+str(self.array[i][j].value)+' '
            s = s+'\n'
        return s

    """
    Textual version of the alignment with *s for inserts and deletes
    """
    def align(self):
        self.findPath()
        return self.getAlignment()
        
    def getAlignment(self):
        path = self.last().getLinks()
        alignment = []
        for l in path:
                if l[3] == 'DELETE':
                    t = (self.v1[l[0]], '*')
                elif l[3] == 'INSERT':
                    t = ('*', self.v2[l[1]])
                else:
                    x = self.v1[l[0]]
                    y = self.v2[l[1]]
                    if l[2] == 0:
                        t = (x, y)
                    else:
                        t = (x, y, l[2])
                alignment.append(t)
        return alignment[1:-1]

def red(s):
    return """<span style="color:red">%s</span>"""%(s)

def html(self):
    self.findPath()
    path = self.last().getLinks()
    s = ""
    for l in path[1:-1]:
        x = self.v1[l[0]]
        y = self.v2[l[1]]
        if l[3] == 'DELETE':
            t = red("%s:%s"%(x, '*'))
        elif l[3] == 'INSERT':
            t = red("%s:%s"%('*', y))
        else:
            t = "%s:%s"%(x, y)
            if not x == y:
                t = red(t)
        if not x == y:
            s += "%s<br>"%(t)
    return s
