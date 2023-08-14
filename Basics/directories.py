from basics.utilities import *

def directories(d=".", n=2, files=False):
    tree = [d.split("/")[-1]]
    for s in sorted(os.listdir(d)):
        if os.path.isdir(os.path.join(d, s)):
            if n > 0:
                tree.append(directories(os.path.join(d, s), n-1, files=files))
        elif files:
            tree.append([s])
    return tree

def showtree(tree, indent="", first=True):
    s = ""
    if first:
        s += "--|%s"%(tree[0])
    else:
        s += "\n%s  |%s"%(indent, tree[0])
    for i, x in enumerate(tree[1:]):
        if len(tree[1:]) > 8 and i > 1:
            s += "\n%s%s..."%(" "*(len(tree[0])+5), indent)
            break
        if i > 0:
            s += "\n%s%s  |"%(" "*(len(tree[0])+3), indent)
        s += showtree(x, indent=indent+" "*(len(tree[0])+3), first=(i==0))
    return s
        
    
    
            
        
    
