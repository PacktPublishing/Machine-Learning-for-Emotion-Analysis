from nltk.parse.malt import MaltParser
from nltk import pos_tag

mp = MaltParser(".", "engmalt.linear-1.7.mco", tagger=pos_tag)

def maltparse(s="are there great tenors who are Italian ."):
    p = mp.parse_one(s.replace(".", ".").replace("  ", " ").split())
    p = p.to_conll(4).strip().split("\n")
    conll = ""
    for i, x in enumerate(p):
        conll += "%s\t%s\n"%(i+1, x)
    return conll

