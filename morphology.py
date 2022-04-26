from utilities import *
from nltk.corpus import wordnet

AZ = re.compile("^[a-z]*$")
ALLWORDS = set(w for w in wordnet.all_lemma_names() if AZ.match(w))

AFFIXES = ["s", "ing", "en", "ed", "er", "est"]

def splitword(form):
    for affix in AFFIXES:
        if form.endswith(affix) and form[:-len(affix)] in ALLWORDS:
            yield "%s+%s"%(form[:-len(affix)], affix)

MCLASSES = {
    "take": {"presentNot3s":"", "present3s":"s", "prespart":"ing", "pastpart":"en"},
    "took": {"past":""},
    "think":  {"presentNot3s":"", "present3s":"s", "prespart":"ing", "pastpart":""},
    "thought":  {"past": "", "pastpart":""}}

def mclass(word):
    try:
        return MCLASSES[word]
    except:
        return {"presentNot3s":"", "present3s":"s", "prespart":"ing", "pastpart":"ed","past":"ed"}

AFFIXES = {
    "": ["presentNot3s", "past"],
    "ing": ["prespart"],
    "s": ["present3s"],
    "ed": ["past", "pastpart"],
    "en": ["pastpart"]}

def match(affix, t1, t2):
    cases = []
    for k in t2:
        if k in t1 and t1[k] == affix:
            cases.append(k)
    return cases

def splitword(form):
    for affix in AFFIXES:
        if affix == "":
            lemma = form
        else:
            lemma = form[:-len(affix)]
        if form.endswith(affix) and lemma in ALLWORDS:
            cases = match(affix, mclass(lemma), AFFIXES[affix])
            if not cases == []:
                yield "%s+%s:%s"%(lemma, affix, cases)

def splitwords(words):
    for word in words:
        for case in splitword(word):
            print(case)

def timeIt(pattern, string, n=10):
    T0 = time.time()
    for i in range(n):
        for m in pattern.finditer(string):
            pass
    return time.time()-T0

FSTS = """
came ==> come + ed
earnt ==> earn + ed
C0 y X0:ing ==> C0 ie + X0
C0 X0:ly ==> C0 l e + X0
i X0:e(d|r|st)|ly ==> y + X0
ie X0:s ==> y + s
X0:((d|g|t)o)|x|s(h|s)|ch es ==> X0 + s
V0 C0 C0 X0:ed|en|ing ==> V0 C0 + X0
V0 C0 X0:ed|en|ing ==> V0 C0 e + X0
C0 h X0:ed|en|ing ==> C0 h e + X0
X0:s|ing|e(|d|n|r|st)|ly ==> + X0
"""

def compileFST(fst):
    [lhs, rhs] = fst.split("==>")
    pattern = ""
    groups = set()
    for x in lhs.strip().split():
        if x[0].isupper():
            c, n = x[0], x[1]
            if c == "V":
                range = "[aeiou]"
            elif c == "C":
                range = "[qwrtypsdfghjklzxcvbnm]"
            elif c == "X" and x[2] == ":":
                range = x[3:]
            else:
                raise Exception("Ill-formed pattern: %s"%(lhs))
            group = "%s%s"%(c, n)
            if group in groups:
                pattern += "(?P=%s)"%(group)
            else:
                groups.add(group)
                pattern += "(?P<%s>%s)"%(group, range)
        else:
            pattern += x
    pattern += "$"
    rep = ""
    for r in rhs.strip().split():
        if r[0].isupper():
            rep += "\g<%s>"%(r)
        else:
            rep += r
    return re.compile(pattern), rep
            
def compileFSTs(fsts=FSTS):
    patterns = []
    for fst in FSTS.strip().split("\n"):
        patterns.append(compileFST(fst))
    return patterns

def applyFSTs(word0, fsts, known={"has": ("have", "s"),
                                  "had": ("have", "ed"),
                                  "be": ("be", ""),
                                  "was": ("be", "ed"),
                                  "were": ("be", "ed"),
                                  "is": ("be", "s"),
                                  "are": ("be", ""),
                                  "am": ("be", ""),
                                  "seen": ("see", "en"),
                                  "said": ("say", "ed"),
                                  "master": ("master", ""),
                                  "brought": ("bring", "ed"),
                                  "thought": ("think", "ed"),
                                  "caught": ("catch", "ed"),
                                  "taught": ("teach", "ed"),
                                  "ate": ("eat", "ed"),
                                  "need": ("need", ""),
                                  "us": ("us", ""),
                                  "her": ("her", ""),
                                  "me": ("me", ""),
                                  "we": ("we", ""),
                                  "his": ("his", ""),
                                  "as": ("as", ""),
                                  "men": ("man", "s"),
                                  "women": ("woman", "s"),
                                  "children": ("child", "s"),
                                  "the": ("the", ""),
                                  "best": ("good", "est"),
                                  "better": ("good", "er"),
                                  "added": ("add", "ed"),
                                  "died": ("die", "ed"),
                                  "dies": ("die", "s"),
                                  "flew": ("fly", "ed"),
                                  "blanche": ("blanche",""),
                                  "pulled": ("pull", "ed"),
                                  "rode": ("ride", "ed"),
                                  "surest": ("sure", "est"),
                                  "surer": ("sure", "er"),
                                  "took": ("take", "ed"),
                                  "king": ("king", ""),
                                  "cooked": ("cook", "ed")}):
    if not word0 in known:
        if "+" in word0:
            known[word0] = (word0, "")
        else:
            for fst, reps in fsts:
                m = fst.search(word0)
                if m:
                    try:
                        [word1, affix] = fst.sub(reps, word0).split("+")
                    except:
                        raise Exception("%s %s"%(word0, m))
                    if word1 in ALLWORDS:
                        known[word0] = (word1, affix)
                        break
            else:
                known[word0] = (word0, "")
    return known[word0]

def root(word, fsts=compileFSTs()):
    return applyFSTs(word, fsts)[0]

def checkAllWords(l, fsts, out, inflected=True, N=500):
    T = time.time()
    n = 0
    with open(out, "w") as out:
        for w in l:
            root, suffix = lookup(w, fsts)
            if inflected and not suffix == "":
                continue
            out.write("%s\t%s\t%s\n"%(w, root, suffix))
            n += 1
            if n > N:
                break
    return time.time()-T

def timeAllWords(l, fsts):
    T = time.time()
    for w in l:
        root, suffix = lookup(w, fsts)
    return time.time()-T
        


 
 
                 



                   
