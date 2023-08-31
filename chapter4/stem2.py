from basics.utilities import *
from nltk.corpus import wordnet
from basics.corpora import *
from .stem1 import allstems, sortstem
"""
Get all the standard words (no upper case, no numbers, no-hyphens) in the wordnet list
"""

def readAllWords():
    return set(wordnet.all_lemma_names())

try:
    ALLWORDS
except:
    ALLWORDS = readAllWords()

"""
The rules: we need to deal with "came" as a rule rather than a special
case because we have to allow for became and overcame, and likewise we
want to allow for "earnt" and "learnt".
"""

SPELLINGRULES = """
imp ==> in + p
V tion ==>  V te + ion
C ation ==> C e + ion
sion ==> de + ion
C y X:ing ==> C ie + X
C X:ly ==> C le + X
C aid ==> C ay + ed
i X1:(e(d|r|st)|ly|ness) ==> y + X1
i e s ==> y + s
X:((d|g|t)o)|x|s(h|s)|ch es ==> X + s
C0 rous ==> C0 e r + ous
C0 i C1 (?!(.*(a|e|i|o|u))) ==> C0 y + C1
X0 (?!(?P=X0)) C X1:e(d|n|r|st)|i(ng|c) (?!(.*(e|i))) ==> X0 C e + X1
^ C V0 X1:e(d|n|r|st)|ing (?!(.*(e|i))) ==> C V0 e + X1
u X1:e(d|n|r|st)|ing ==> ue + X1
C0 V C1 C1 X:(e(d|n|r|st)|ing) ==> C0 V C1 + X
"""

"""
Compile a rule like 

   ee X0 (d|n) ==> ee + e X0

to a regex and a substitution like

  (re.compile('ee(?P<X0>(d|n))$'), 'ee+e\\g<X0>')

The pattern will match a sequence like "een" or "eed" at the end of a word, and will bind the group X0 to "n" (or "d")). The substition group will then convert "seen" to "see+en" or "freed" to "free+ed"

"""
def compileFST(fst, i):
    [lhs, rhs] = fst.split("==>")
    pattern = ""
    vpattern = re.compile("(?P<var>[A-Z]*)(?P<N>(\d*))(:(?P<range>\S+))?")
    groups = set()
    lhs = (lhs+" K").replace("$", " $")
    for x in lhs.strip().split():
        """ 
        Upper cases elements are named groups
        """
        if x[0].isupper():
            m = vpattern.match(x)
            c = m.group("var")
            n = "%s_%s"%(m.group("N"), i)
            range = m.group("range")
            """V0 will match any vowel and bind to V0"""
            if c == "V":
                range = "[aeiou]"
                """C0 will match any consonant and bind to C0"""
            elif c == "C":
                range = "[qwrtypsdfghjklzxcvbnm]"
                """ 
                X0:(q|w|e) will match any of "q", "w" or "e" and bind
                it to X0. You can use this to define new classes on
                the fly, e.g. x|s(h|s)|ch
                """ 
            elif c == "K":
                range = "[a-z]*"
            else:
                try:
                    range = vpattern.match(x).group("range")
                    if not range:
                        range = "[a-z]"
                except:
                    raise Exception("Ill-formed pattern: %s"%(lhs))
            group = "%s%s"%(c, n)
            """
            The first time you use a group in a pattern you set it,
            subsequent times you have to match it
            """
            if group in groups:
                pattern += "(?P=%s)"%(group)
            else:
                groups.add(group)
                pattern += "(?P<%s>%s)"%(group, range)
        else:
            pattern += re.compile("(?P<V>(X|C|V)\d*)").sub("\g<V>_%s"%(i), x)
    rep = ""
    for r in (rhs.strip()+" K").split():
        if r[0].isupper():
            rep += "\g<%s_%s>"%(r, i)
        else:
            rep += r
    return pattern, rep

"""
Compile your list of rules into a list of rules. It might be slightly
quicker to compile the whole list of rules into a single disjunctive
pattern, but it makes looking after the bindings more complex and it
makes it hard to backtrack if you found something that matched but
there was no such root.
"""
def compileFSTs(fsts=SPELLINGRULES):
    patterns = []
    for i, fst in enumerate(fsts.replace("\n\n", "\n").strip().split("\n")):
        patterns.append(compileFST(fst, i))
    pattern = re.compile("^(%s).*"%("|".join(p[0] for p in patterns)))
    return pattern, [p[1] for p in patterns]

FSTS = compileFSTs(SPELLINGRULES)

def applyFSTs(form, fsts, n=0):
    m = fsts[0].search(form)
    if m:
        k = m.group(0)
        g = m.groupdict()
        for x in g:
            if not g[x] is None:
                k1 = fsts[0].sub(fsts[1][int(x.split("_")[-1])], k)
                return form[:-len(k)]+k1

    
PREFIXES = {"un", "dis", "re", "de", "in"}
SUFFIXES = {"ing", "s", "ed", "en", "er", "est", "ly", "ion", "ness", "ic", "al"}
SUFFIXFSTS = re.compile("^(%s)"%("|".join(SUFFIXES)))

def stemhelper(left, right, p, r, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
    if left == "" and right == "":
        yield "%s%s%s"%(p, r, s)
    if not right == "":
        if right[0] == "+":
            right = right[1:]
        else:
            # If it wasn't forced to be a morpheme boundary by a spelling rule, try moving along the string
            for y in stemhelper(left+right[0], right[1:], p, r, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
                yield y
            x = applyFSTs(right, fsts)
            if x:
                x = x.split("+")
                for y in stemhelper(left+x[0], "+%s"%(x[1]), p, r, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
                    yield y
    if r:
        if left in suffixes:
            for y in stemhelper("", right, p, r, "%s+%s"%(s, left), prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
                yield y
    else:
        if left in allwords:
            if right == "" or SUFFIXFSTS.match(right):
                for y in stemhelper("", right, p, left, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
                    yield y
        if left in prefixes:
            for y in stemhelper("", right, "%s%s-"%(p, left), r, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS):
                yield y
                
def stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    return stemhelper("", form, "", "", "", prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES)


def allstems(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    return sorted(stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES), key=sortstem)
