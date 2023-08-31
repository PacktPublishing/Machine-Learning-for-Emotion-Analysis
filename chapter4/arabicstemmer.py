#!/usr/local/anaconda3/bin/python3

import sys, os

import re, sys
from basics.utilities import *
from basics.a2bw import convert, a2bwtable, bw2atable
from basics import corpora

import datetime

"""
(?P<name>xya*b) will match any sequence like "xyaaaab" and call it "name"

(?(name)(abcd)|(pqrs)): if some group called "name" has already been
matched, then this will match "abcd", otherwise it will match
"pqrs". This is fancy, and means that these things are not really
regexes and therefore matching is not linear time
(https://swtch.com/~rsc/regexp/) (well actually the versions called
NROOT-OLD and VROOT-OLD could take an * extremely* long time -- I've
cut them back to the simpler NROOT and VROOT, but I suspect that I may
now be missing something).
"""
rep = re.compile("(?P<x>.)(?P=x)+")

patterns = {"CLOSED": """(?P<NCPREP>b|fy|AlA|mn)? PRON? (?<=.)|(?P<PARTICLE>An)|(?P<EMOJI>[^؟-ۼa-zA-Z0-9\.!%&\?'"-,]+)""",
            "LV": "(A|w|y)(?!(A|w|y))",
            "SV": "i|o|u|a",
            "V": "#SV? (#SV| #LV)",
            "CA": "[^AiouapF]",
            "CB": "[^iouapFy]",
            "X": "[^AiouapFwy]",
            "RC1": "(?P<C1>#CA)(?P=C1)*?",
            "RC2": "(?P<C2>#CA)(?P=C2)*?",
            
            "VROOT": "#RC1 #CB #LV? (?P<G2>#RC2)?",
            "NROOT": "(?P<NAME>Allh?) | #RC1 #LV? (#CB #LV?){1,2}? (?P<G2>#RC2)?",
            
            "NDERIV": "mu?(?=(#X #V?){3,})|A(?=(.t.A))|A(s+)t|mst|Ist|I(?=..A)",
            "ADJ": "y",
            "NOUN": "(NDERIV? NROOT ADJ?)",
            
            "VDERIV": "t(u|.A)|A(?=[^s].t)|(?<=(y|n|t))st|(?<!(y|t|n))Ast|An",
            "VSTEM": "#VDERIV? #VROOT",
            
            "XSTEM": "(#CA #V?){3,}",
            "NCONJ": "(w|f)(?=(Al|#XSTEM))",
            "VCONJ": "(w|f) (?=#XSTEM)",
            "NNEG": "lA(?=..)",
            "NEG": "(m|l)A(?=...)",
            "INT": "O",
            "PREP": "(ba?|k(?:i?)|l)",
            "PREP1": "#PREP (?=(#DET|(#CA #V?){3,}))",
            "DET": "Al(?!lh?)",
            "PRON": "k(?:u?mA?|n)?|h(u?m)?(?:A*)|hn|nA|y",
            "XXX": "k(?:u?mA?|n)?|h(u?m)?(?:A*)|hn|nA|y",
            "PRON1": "#PRON|#NY",
            "NY": "ny",
            
            "FUT": "s|H|h|b",
            "PRES": "((?P<ON>O|(n(?=[^Awy]{3})))|(?P<Y>y)|(?P<T>t))(a|u)?",
            "PV": "(?P<PAST>)",
            "IMPER": "(?P<IMP>A)",
            "TENSE": "(FUT? PRES)|PV",
            
            "AGR_ON": "",
            "AGR_A": "(?<=(.{3}))(?<!(?:n|h))A",
            "AGR_Y": "(?:An|#AGR_A|w(A|n)|n|)",
            "AGR_T": "(?:yn|An|wn|n|)",
            "AGR_PAST": "t(mA?|n?)|#AGR_A|wA|nA?|",
            "AGR_IMPER": "(#AGR_A|wA|n|)",
            "PERSON": "(?(ON)AGR_ON|(?(Y)AGR_Y|(?(T)AGR_T|(?(PAST)AGR_PAST|(?(IMP)AGR_IMPER)))))",

            "VOC": "yA",
            
            # "AGREE": "tAn|wn|yn|p|h|An|At|", 
            # "An" removed because it introduces too many false positives 
            # (people don't use duals in tweets)
            # Re-added, but only for words with tri-literal stems
            "AGREE": "tAn|wn|yn|a?p|t(?=.)|At|\|t|(?(G2)An)|", 
            
            "CASE": "SV | AF | FA? ",
            "ALLAH": "A?llh?",
            
            "NN": "NCONJ? VOC? NNEG? PREP1? (DET? NOUN AGREE) ((?(G2) (ALLAH|PRON|CASE)?)|(?(NAME) (XXX)?))",
            "VB": "VCONJ? NEG? (IMPER|(INT? TENSE)) VSTEM PERSON (?(G2)(PRON1)?)"}

"""
For each pattern, replace anything that is in uppercase, possibly with digits,
and is in the set of patterns, by the expansion of its value in the set, i.e.
replace "CONJ" in "(CONJ)?(DEFDET)?(STEM)(AGREE)?(PRON)?" by "w|B". Do this
recursively (e.g. VERBs contain TENSEs, but TENSEs contain FUTs and TNS1s)
"""
def expandpattern(p, patterns=patterns, expanded=False):
    if expanded == False:
        expanded = {}
    for i in re.compile("(?P<hash>#?)(?P<name>([A-Z0-9]|_)+)\s*").finditer(p):
        name = i.group("name")
        hash = i.group("hash")
        wholething = i.group(0)
        if name in patterns:
            if hash == "":
                p = p.replace(wholething, "(?P<%s>%s)"%(name, expandpattern(patterns[name], patterns, expanded)))
            else:
                p = p.replace(wholething, "(%s)"%(expandpattern(patterns[name], patterns, expanded)))
            expanded[name] = p
    return p

"""
Do that for all your patterns.
Once you've done it, replace the values by compiled regexes that have ^ and $
to make sure that they match the whole string
"""
def expandpatterns(patterns=patterns):
    expanded = {}
    epatterns = {p:expandpattern(patterns[p], patterns, expanded) for p in patterns}
    for p in epatterns:
        try:
            epatterns[p] = re.compile("^("+epatterns[p].replace(" ", "")+")$")
        except Exception as e:
            pass
    return epatterns

        
EXPANDEDPATTERNS = expandpatterns()
"""
try:
    ARABICPATTERNS = expandpatterns(arabisepatterns())
except:
    print("Couldn't do arabisepatterns")
"""

STANDARDGROUPS = {"NN": ["NCONJ", "VOC", "NNEG", "PREP1", "NAME", "DET", "AGREE", "ALLAH", "CASE", "ADJ", "PRON", "XXX", "NOUN"], 
                  "VB": ["VCONJ", "NEG", "IMPER", "INT", "FUT", "PRES", "PV", "VSTEM", "PERSON", "PRON1"],
                  "CLOSED": ["NCPREP", "PRON", "EMOJI", "ALLAH", "PARTICLE"],}

BASICGROUPS = {"NN": ["NCONJ", "NNEG", "PREP1", "DET", "NOUN", "ALLAH", "PRON", "XXX"], 
                  "VB": ["VCONJ", "NEG", "VSTEM", "PRON1"],
                  "CLOSED": ["NCPREP", "PRON", "EMOJI", "ALLAH", "PARTICLE"],}

TWEETGROUPS= {"NN": ["NOUN", "ALLAH"], 
              "VB": ["VSTEM"],
              "CLOSED": ["NCPREP", "PRON", "EMOJI", "ALLAH", "PARTICLE"],}
        
def lookupword(w, tag, expandedpatterns=EXPANDEDPATTERNS, tags=STANDARDGROUPS, N = False, arabic=True, debugging=False):
    if N:
        T0 = datetime.datetime.now()
        for i in range(N):
             lookupword(w, tag, expandedpatterns, tags=tags, arabic=arabic, debugging=debugging)
        T1 = datetime.datetime.now()
        print("%dK words/second"%(N/((T1-T0).total_seconds()*1000)))
    else:
        if arabic:
            w = convert(w, a2bwtable)
        m = expandedpatterns[tag].match(w)
        if tags:
            try:
                g = m.groupdict()
                if debugging: print(g)
                groups = [(m.start(x), x, g[x]) for x in tags[tag] if x in g and not g[x] == None]
                groups.sort()
                return [(x[1], x[2]) for x in groups]
            except:
                return False
        else:
            return m

def stem(word, tags=STANDARDGROUPS, cnv=lambda x: x, debugging=False):
    word = convert(word, a2bwtable)
    wnoun = lookupword(word, "NN", tags=tags, debugging=debugging)
    wverb = lookupword(word, "VB", tags=tags, debugging=debugging)
    wclosed = lookupword(word, "CLOSED", tags=tags, debugging=debugging)
    if wclosed:
        w = wclosed
    elif wnoun and wverb:
        nparts = [x[0] for x in wnoun if not x[1] == ""]
        vparts = [x[0] for x in wverb if not x[1] == ""]
        if "ALLAH" in nparts or len(vparts) <= len(nparts):
            w = wnoun
        else:
            w = wverb
    elif wnoun:
        w = wnoun
    elif wverb:
        w = wverb
    else:
        return [word]
    return ["%s:%s"%(cnv(x[1]), x[0]) for x in w if not x[1] == ""]

def stemAll(text0, tags=BASICGROUPS):
    text1 = []
    for w in text0:
        text1 += stem(w, tags=tags)
    return text1

def stemTweets(N=10, tfile="CORPORA/TWEETS/IMAN/0K-40K.csv", out=sys.stdout, arabic=False):
    if N < 0:
        N = sys.maxsize
    if not out == sys.stdout:
        out = open(out, "w")
    if arabic:
        cnv = lambda x: convert(x, bw2atable)
    else:
        cnv = lambda x: convert(x, a2bwtable)
    for i, u in enumerate(open(tfile)):
        if i > N:
            return
        if i > 0:
            u = u.split("\t")[1].replace('"""', '').strip()
            # out.write("\n****\t%s\n"%(cnv(u)))
            out.write("\n")
            for t in corpora.tokenise(u, corpora.ARABICPATTERN):
                out.write("%s\t%s\n"%(cnv(t), stemBest(t, cnv=cnv)))
    if not out == sys.stdout:
        out.close()

if "stemmer.py" in sys.argv[0]:
    stemTweets(N=int(checkArgs("N", sys.argv[1:], 10)),
               tfile=checkArgs("tfile", sys.argv[1:], "CORPORA/TWEETS/IMAN/0K-40K.csv"),
               arabic=int(checkArgs("arabic", sys.argv[1:], 1)))
    
