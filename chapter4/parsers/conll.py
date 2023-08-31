from trees import *

def table2list(tableandwords, top=None, parents=None, role="ROOT"):
    table, words = tableandwords
    if top is None:
        top = table[0][0][0]
        parents = []
    l = "%s:%s"%(words[top], role)
    if top in table:
        dtrs = []
        for d in table[top]:
            if not d[0] in parents:
                dtrs.append(table2list((table, words), top=d[0], role=d[1], parents=parents+[d[0]]))
    else:
        dtrs = []
    return [l, dtrs]

def pretty(l, first=True, indent=""):
    if first:
        if indent == "":
            s = l[0]
            n = 0
        else:
            s = "|--%s"%(l[0])
            n = 3
    else:
        n = 3
        s = """
%s|
%s|--%s"""%(indent, indent, l[0])
    for i, d in enumerate(l[1]):
        s = s+pretty(d, first=(i==0), indent=indent+" "*(len(l[0])+n))
    return s

def conll2table(s):
    words = {0:"root"}
    dtrs = {}
    for l in s.strip().split("\n"):
        dtrIndex, dtrWord, tag, headIndex, label = l.split("\t")
        dtrIndex = int(dtrIndex)
        headIndex = int(headIndex)
        words[dtrIndex] = dtrWord
        try:
            dtrs[headIndex].append((dtrIndex, label))
        except:
            dtrs[headIndex] = [(dtrIndex, label)]
    return (dtrs, words)

def conll2tags(s):
    tags = ""
    for l in s.strip().split("\n"):
        dtrIndex, dtrWord, tag, headIndex, label = l.split("\t")
        tags += "%s:%s, "%(dtrWord, tag)
    return tags

def plotconll(conll):
    show(width(table2list(conll2table(conll))))

def parse(text, parser):
    plotconll(parser(text))
