from utilities import *
from nltk.corpus import wordnet

PREFIXES = {"", "un", "dis", "re"}
SUFFIXES = {"", "ing", "s", "ed", "en", "er", "est", "ly", "ion"}
PATTERN = re.compile("(?P<form>[a-z]{3,}) (?P<pos>n|v|r|a) ")

def readAllWords():
    return set(wordnet.all_lemma_names())

try:
    ALLWORDS
except:
    ALLWORDS = readAllWords()
    
def stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    for i in range(len(form)):
        if form[:i] in prefixes:
            for j in range(i+1, len(form)+1):
                if form[i:j] in allwords:
                    if form[j:] in suffixes:
                        yield ("%s-%s+%s"%(form[:i], form[i:j], form[j:])).strip("+-")

ROOTPATTERN = re.compile("^(.*-)?(?P<root>.*?)(\+.*)?$")
def sortstem(w):
    r = ROOTPATTERN.match(w).group("root")
    return (len(r), r)

def allstems(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES):
    return sorted(stem(form, prefixes=PREFIXES, allwords=ALLWORDS, suffixes=SUFFIXES), key=sortstem)
