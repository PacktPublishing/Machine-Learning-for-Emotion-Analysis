import matplotlib.pyplot as plt

def plotpoints(plt, points, colour="black", linestyle="-", linewidth=1):
    plt.plot([point[0] for point in points],
             [point[1] for point in points],
             color=colour, linewidth=linewidth, linestyle=linestyle)

"""
>>> words, dtrs = sdpparse("John loves a woman")
>>> tree = table2list(dtrs, words)
>>> tree
['loves', [['nsubj', ['John', []]], 
           ['obj', ['woman', [['det', ['a', []]]]]]]]

"""
    
def width(tree, depth=0):
    dtrs = []
    hd = tree[0]
    for i, d in enumerate(tree[1]):
        d = width(d, depth+8)
        dtrs.append(d)
    s = sum(d[1] for d in dtrs)
    w = max(len(hd)*0.5, s)
    return [hd, w, depth, dtrs]

def maxdepth(l, i=0):
    i = l[2]
    for d in l[3]:
        i = max(i, maxdepth(d))
    return i

def show(l, parent=False, offset=0, ylim=0):
    if not parent:
        fig = plt.figure()
        plt.axis("off")
        ylim = maxdepth(l)
        fig.set_size_inches(6, 4)
        plt.xlim([0, l[1]])
        plt.ylim([0, ylim])
    plt.text(offset+l[1]*0.25, ylim-l[2], l[0])
    if parent:
        points = [(offset+l[1]*0.25+len(l[0])*0.1, ylim-l[2]+1.3), parent]
        plotpoints(plt, points)
    offset0 = offset
    for d in l[3]:
        show(d, offset=offset, parent=(offset0+l[1]*0.25+len(l[0])*0.1, ylim-l[2]-0.8), ylim=ylim)
        offset += d[1]
    if not parent:
        plt.show()

"""
['1', 7, 3.5, 0, 
     ['2', 7, 3.5, 5, 
         ['3', 5, 2.5, 10, 
               ['abcde', 5, 2.5, 15]], 
         ['4', 2, 6.0, 10, 
               ['5', 1, 5.5, 15], 
               ['6', 1, 6.5, 15]]]]
"""
    

    
            
    
        
        
