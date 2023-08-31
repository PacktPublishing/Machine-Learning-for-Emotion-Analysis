from basics.utilities import *
from nltk.corpus import wordnet

PREFIXES = {"", "un", "dis", "re", "in"}
SUFFIXES = {"", "ing", "s", "ed", "en", "er", "est", "ly", "ion"}

def readAllWords():
    return set(wordnet.all_lemma_names())

try:
    ALLWORDS
except:
    ALLWORDS = readAllWords()

def stem0(form, prefixes=PREFIXES, suffixes=SUFFIXES):
    for prefix in prefixes:
        if form.startswith(prefix):
            form1 = form[len(prefix):]
            for suffix in suffixes:
                if form1.endswith(suffix):
                    if suffix == "":
                        yield (prefix, form1, "")
                    else:
                        yield (prefix, form1[:-len(suffix)], suffix)
    
def stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    for i in range(len(form)):
        if form[:i] in prefixes:
            for j in range(i+1, len(form)+1):
                if form[i:j] in allwords:
                    if form[j:] in suffixes:
                        yield form[:i], form[i:j], form[j:]

ROOTPATTERN = re.compile("^(.*-)?(?P<root>.*?)(\+.*)?$")
def sortstem(w):
    r = ROOTPATTERN.match(w).group("root")
    return (len(r), r)

def allstems(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    return sorted(stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES), key=sortstem)
