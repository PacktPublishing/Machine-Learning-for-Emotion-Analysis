from basics.utilities import *
from nltk.corpus import wordnet
from corpora import *
from .stem2 import applyFSTs, FSTS
    
def readTerm(s):
    s = s.strip()
    p0 = re.compile("(?P<g>->|<-|[a-z]*)(?P<rest>.*)")
    if s[0] == "(":
        terms, s = readTerms(s[1:])
        if not s[0] == ")":
            raise Exception("Closing bracket expected: %s"%(s))
        return tuple(terms), s[1:]
    else:
        m = p0.match(s)
        r = m.group("rest")
        if len(r) > 0 and r[0] == "[":
            r = r[1:].split("]", 1)
            d = {"hd":m.group("g")}
            for i in re.compile("(?P<key>\S*)\s*=\s*(?P<value>.*?)($|,)").finditer(r[0]):
                d[i.group("key")] = i.group("value")
            return d, r[1]
        else:
            return m.group("g"), r

def readTerms(s):
    terms = []
    while s and not s[0] == ")":
        term, s = readTerm(s)
        terms.append(term)
    terms = tuple(terms)
    if len(terms) == 1:
        terms = terms[0]
    return terms, s

ROOTS = {"v": "v->tns",
         "n": "n->num",
         "a": "a->cmp",
         "r": "r"}

def readAllWords(roots=ROOTS):
    allwords = {}
    for w in wordnet.all_lemma_names():
        if not w in allwords:
            allwords[w] = set()
        for s in wordnet.lemmas(w):
            allwords[w].add(s._synset._pos)
    for w in allwords:
        l = allwords[w]
        allwords[w] = []
        for x in l:
            if x in roots:
                allwords[w].append(readTerms(roots[x])[0])
    return allwords

ALLWORDS = readAllWords()

def fixaffixes(affixes):
    p = re.compile("\s*;\s*")
    return {a: [readTerms(s)[0] for s in p.split(affixes[a].strip())] for a in affixes}

PREFIXES = fixaffixes(
    {"un": "((a->cmp)->tns)->(v->tns)",
     "re": "(v->tns)->(v->tns)",
     "dis": "(v->tns)->(v->tns)"})

SUFFIXES = fixaffixes(
    {
        # inflectional suffixes
        "": "tns; num; cmp",
        "ing": "tns; (a<-(v->tns))",
        "ed": "tns; (a<-(v->tns))",
        "s": "tns; num",
        "en": "tns",
        "est": "cmp",

        # derivational suffixes
        "ly": "r<-a",
        "ic": "a<-(n->num)",
        "al": "a<-a",
        "er": "(n->num)<-(v->tns); cmp",
        "ion": "(n->num)<-(v->tns)",
        "ment": "(n->num)<-(v->tns)",
        "ous": "a<-(n->num)",
        "less": "a<-(n->num)",
        "ness": "(n->num)<-(v->tns)",
        "able": "a<-(v->tns)",
     })

def combine(l0, l1):
    combinations = []
    for x0 in l0:
        for x1 in l1:
            if isinstance(x0, tuple) and x0[1] == "->" and x0[2] == x1:
                combinations.append(x0[0])
            if isinstance(x1, tuple) and x1[1] == "<-" and x1[2] == x0:
                combinations.append(x1[0])
    return combinations

SUFFIXFSTS = re.compile("^(%s)"%("|".join(SUFFIXES)))

def stemhelper(left, right, p, r, s, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, fsts=FSTS, combine=combine):
    if left == "" and right == "":
        yield r
    if not right == "":
        if right[0] == "+":
            right = right[1:]
        else:
            # If it wasn't forced to be a morpheme boundary by a spelling rule, try moving along the string
            for y in stemhelper(left+right[0], right[1:], p, r, s, prefixes=prefixes, allwords=allwords, suffixes=suffixes, fsts=FSTS, combine=combine):
                yield y
            x = applyFSTs(right, fsts)
            if x:
                x = x.split("+")
                for y in stemhelper(left+x[0], "+%s"%(x[1]), p, r, s, prefixes=prefixes, allwords=allwords, suffixes=suffixes, fsts=FSTS, combine=combine):
                    yield y
    if r:
        if left in suffixes:
            combinations = combine(r[1], suffixes[left])
            if not combinations == []:
                for y in stemhelper("", right, p, ("%s+%s"%(r[0], left), combinations), s+[[left, suffixes[left]]], prefixes=prefixes, allwords=allwords, suffixes=suffixes, fsts=FSTS, combine=combine):
                    yield y
    else:
        if left in allwords:
            if right == "" or SUFFIXFSTS.match(right):
                combinations = allwords[left]
                pp = ""
                for prefix in p:
                    pp = "%s-%s"%(prefix[0], pp)
                    combinations = combine(prefix[1], combinations)
                for y in stemhelper("", right, p, [pp+left, combinations], s, prefixes=[], allwords=allwords, suffixes=suffixes, fsts=FSTS, combine=combine):
                    yield y
        if left in prefixes:
            for y in stemhelper("", right, [[left, prefixes[left]]]+p, r, s, prefixes=prefixes, allwords=allwords, suffixes=suffixes, fsts=FSTS, combine=combine):
                yield y
import inspect
ROOTPATTERN = re.compile("^(.*-)?(?P<root>.*?)(\+.*)?$")
def sortstem(w):
    return [len(ROOTPATTERN.match(w[0]).group("root")), len(list(x for x in w[1] if isinstance(x, tuple)))]

def stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES, combine=combine):
    return stemhelper("", form, [], [], [], prefixes=prefixes, allwords=allwords, suffixes=suffixes, combine=combine)

def allstems(form, module):
    prefixes=module.PREFIXES; allwords=module.ALLWORDS; suffixes=module.SUFFIXES; combine=module.combine
    return sorted(stem(form, prefixes=prefixes, allwords=allwords, suffixes=suffixes, combine=combine), key=sortstem)
