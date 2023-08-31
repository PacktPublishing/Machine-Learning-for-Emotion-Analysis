import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy
from utilities import *

def plotpoints(plt, points, colour="black", linestyle="-", linewidth=1):
    plt.plot([point[0] for point in points],
             [point[1] for point in points],
             color=colour, linewidth=linewidth, linestyle=linestyle)

"""
Stuff for find convex hulls adapted from
https://www.geeksforgeeks.org/convex-hull-using-jarvis-algorithm-or-wrapping/
"""

def ccw(a, b, c):
    return not a == b and (b[1]-a[1])*(c[0]-b[0]) < (b[0]-a[0])*(c[1]-b[1])

def naiveCH(points, I=sys.maxsize):
    # There must be at least 3 points
    n = len(points)
    if len(points) < 3:
        return
    points = points.tolist()
    """
    Sort points by (X, Y). Then position of leftmost (lowest
    to break ties) in the sorted list is 0
    """
    points.sort()
    p = 0
    hull = []
    """
    Start from leftmost point, keep moving counterclockwise
    until reach the start point again. This loop runs O(h)
    times where h is number of points in result or output.
	"""
    while(len(hull) < I):
        # Add current point to result
        hull.append(points[p])
        """
		Search for a point 'q' such that orientation(p, q,
		x) is counterclockwise for all points 'x'. The idea
		is to keep track of last visited most counterclock-
		wise point in q. If any point 'i' is more counterclock-
		wise than q, then update q. q is the next position, except
        that if it's the last position then it's reset to 0
        """
        q = (p + 1) % n

        for i, k in enumerate(points):
			# If i is more counterclockwise
			# than current q, then update q
            if(ccw(points[p], points[i], points[q])):
                q = i;

        """
		Now q is the most counterclockwise with respect to p
		Set p as q for next iteration, so that q is added to
		result 'hull'
		"""
        p = q

		# Terminate when you get back to that start and close the loop
        if(p == 0):
            hull.append(points[0])
            break
    return hull

def naiveCHdemo(I):
    X, data = make_blobs(n_samples=40, centers=2, random_state=6)
    fig = plt.figure()
    fig.set_size_inches(3.14, 2.4)
    xlim = min(X[:, 0])-1, max(X[:, 0])+1
    ylim = min(X[:, 1])-1, max(X[:, 1])+1
    ax = plt.axes()
    # ax.set_aspect("equal",adjustable='box')
    plt.axis("on")
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.scatter(X[:, 0], X[:, 1], color="black", s=30, marker="$R$")
    X0, X1 = split(X, data)
    for x, y in X0:
        plt.text(x, y, "B", color="blue", fontsize="x-small", fontweight="bold")
    for x, y in X1:
        plt.text(x, y, "R", color="red", fontsize="x-small", fontweight="bold")
    X0 = numpy.array(X0)
    plotpoints(plt, naiveCH(X0, I), linestyle=(0, (3, 0)), colour="blue", linewidth=1)
    plt.show()
                
COLOURS = ["blue", "red"]
LINESTYLES = ["solid", (0, (1, 3))]

def convexhull(points, colour="0", linestyle="solid", N=sys.maxsize):
    hull = naiveCH(points)
    pts0 = [h[0] for h in hull[:N]]
    pts1 = [h[1] for h in hull[:N]]
    if N == sys.maxsize:
        pts0, pts1 = pts0+[pts0[0]], pts1+[pts1[0]]
    return pts0, pts1

def equn(pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1
    a = (y0-y1)/(x0-x1)
    b = y1-a*x1
    return a, b

def dist(pt1, pt2):
    return numpy.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def shortest(p, l):
    p0, p1 = l
    a0, b0 = equn(p0, p1)
    a1 = -1/a0
    b1 = p[1]-a1*p[0]
    x = (b1-b0)/(a0-a1)
    y = a1*(x)+b1
    if p0[0] <= x <= p1[0]:
        return (p, (x, y)), dist(p, (x, y))
    else:
        return (0, 0), sys.maxsize
    
def nearest(pts1, pts2):
    all = []
    for pt1 in pts1:
        for pt2 in pts2:
            all.append([dist(pt1, pt2), pt1, pt2])
    all.sort()
    x0 = all[0]
    for x1 in all[1:]:
        if not (x0[1] == x1[1] or x0[2] == x1[2]):
            break
    return x0[1:], x1[1:]

def midpt(pt1, pt2):
    return ((pt1[0]+pt2[0])/2, (pt1[1]+pt2[1])/2)

def CoM(points):
    return sum(points[:, 0])/len(points), sum(points[:, 1])/len(points)

def split(X, data):
    X0 = []
    X1 = []
    for x, d in zip(X, data):
        if d == 0:
            X0.append(x)
        elif d == 1:
            X1.append(x)
    return X0, X1

def convexhulldemo(switchpoints=0, switchlist=[], ignore=[], I=0, numbered=False, N=sys.maxsize):
    X, data = make_blobs(n_samples=40, centers=2, random_state=6)
    fig = plt.figure()
    fig.set_size_inches(3.14, 2.4)
    xlim = min(X[:, 0])-1, max(X[:, 0])+1
    ylim = min(X[:, 1])-1, max(X[:, 1])+1
    ax = plt.axes()
    # ax.set_aspect("equal",adjustable='box')
    plt.axis("on")
    plt.xlim(xlim)
    plt.ylim(ylim)
    if numbered:
        for i, (x, y) in enumerate(X):
            plt.text(x+0.1, y, i)
    X = numpy.array([(x, y, i) for i, (x, y) in enumerate(X)])
    S0, S1 = split(X, data)
    X0 = []
    X1 = []
    ignorelist = []
    for p in S0:
        (x, y, i) = p
        if i in ignore:
            ignorelist.append(p)
        elif i in switchlist:
            X1.append(p)
        else:
            X0.append(p)
    for p in S1:
        (x, y, i) = p
        if i in ignore:
            ignorelist.append(p)
        elif i in switchlist:
            X0.append(p)
        else:
            X1.append(p)
    for i in ignorelist:
        plt.text(i[0], i[1], "X", color="brown", fontsize="x-small", fontweight="bold")
    X0 = numpy.array(X0)
    X1 = numpy.array(X1)
    xx = sorted(X[:, 0])
    leastX, mostX = xx[0], xx[-1]
    for x, y, i in X0:
        plt.text(x, y, "B", color="blue", fontsize="x-small", fontweight="bold")
    for x, y, i in X1:
        plt.text(x, y, "R", color="red", fontsize="x-small", fontweight="bold")
    if I > 0:
        pts0, pts1 = convexhull(X0, N=N)
        points0 = list(zip(pts0, pts1))
        plotpoints(plt, points0, colour=(0.6, 0.6, 0.6, 0.5), linestyle="solid")
        pts0, pts1 = convexhull(X1)
        points1 = list(zip(pts0, pts1))
        if N == sys.maxsize:
            plotpoints(plt, points1, colour=(0.6, 0.6, 0.6, 0.5), linestyle="solid")
        l0, l1 = nearest(points0, points1)
        l0, l1 = sorted([l0[0], l1[0]]), sorted([l0[1], l1[1]])
    if I == 1.5:
        com0 = CoM(X0)
        circle1 = plt.Circle((com0[0], com0[1]), 0.3, color='black')
        plt.gca().add_patch(circle1)
        com1 = CoM(X1)
        circle1 = plt.Circle((com1[0], com1[1]), 0.3, color='black')
        plt.gca().add_patch(circle1)
    if I > 1.5:
        plotpoints(plt, l0, colour=(1, 0.5, 0, 0.5), linestyle=(0, (2, 2)), linewidth=3)
        plotpoints(plt, l1, colour=(0, 1, 0, 0.5), linestyle=(0, (0.5, 0.5)), linewidth=3)
    if I == 2:
        a0, c0 = equn(l0[0], l0[1])
        plotpoints(plt, [(leastX, a0*leastX+c0), (mostX, a0*mostX+c0)], colour=(1, 0.5, 0, 0.5), linestyle=(0, (2, 2)), linewidth=3)
        a0, c0 = equn(l1[0], l1[1])
        plotpoints(plt, [(leastX, a0*leastX+c0), (mostX, a0*mostX+c0)], colour=(0, 1, 0, 0.5), linestyle=(0, (0.5, 0.5)), linewidth=3)
    if I > 2:
        D = sys.maxsize
        for p in l0:
            (p, q), d = shortest(p, l1)
            if d < D:
                D = d
                (s0, s1) = (p, q)
        for p in l1:
            (p, q), d = shortest(p, l0)
            if d < D:
                D = d
                (s0, s1) = (p, q)
        plotpoints(plt, [s0, s1], linestyle=(0, (1,2)))
        m = midpt(s0, s1)
        a0, c0 = equn(s0[:2], s1[:2])
        a1 = -1/a0
        c1 = m[1]-a1*m[0]
        plotpoints(plt, [(leastX, a1*leastX+c1), (mostX, a1*mostX+c1)])
    plt.show()
    
def pqr():
    plt.axis("off")
    plt.gca().set_aspect("equal")
    points = [(0, 0), (1, 1), (1, 2)]
    for l, p in zip(["A", "B", "C"], points):
        plt.text(p[0], p[1]+0.1, l)
    plotpoints(plt, points[:-1])
    plotpoints(plt, points[1:])
    points1 = [(3, 0), (4, 1), (5, 1)]
    for l, p in zip(["A'", "B'", "C'"], points1):
        plt.text(p[0], p[1]+0.1, l)
    plotpoints(plt, points1[:-1])
    plotpoints(plt, points1[1:])
    plt.show()
               
