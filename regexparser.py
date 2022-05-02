from utilities import *
from corpora import *
from nltk.corpus import wordnet
from morphology import root

"""
Use regexes to find specified dependency pairs by looking for examples of
patterns like "a main verb can be followed by a sequence of noun
modifiers followed by the head noun of its object". This doesn't give
you a full parse tree, but it can provide useful information and it is
very fast and quite easy to extend.

We define a set of tags, which map groups of actual tags to more
general cases (e.g. that a determiner could be any of DT (determiner),
AT (article), CR (cardinal number), DP (possessive pronoun) -- the BNC
distinguishes between these, but for current purposes they are all
equivalent).

We then use these inside rules for matching pairs of words. The rules
are expanded by replacing tags by their corresponding patterns, so
that (?P<TVERB>verb) det? nmod* (?P<OBJ>noun) (which says that you
could have a verb labelled TVERB followed by an optional determiner
and a possibly empty sequence of noun modifiers and a noun labelled
OBJ) becomes a regular expression like

>>> print(expandPatterns(OBJRULES, makeTagTable()).pattern)
(?P<TVERB>\s*(\S*:VV|VH))\s*(\S*:(DT|AT|CR))?\s*(\S*:(AJ|NN))*(?P<OBJ>\s*(\S*:NN))

This regex is fairly unreadable, but it can be applied very fast and
will identify a large proportion of the occurrences of verbs with
standard NPs as objects (it will also pick up cases like "I saw the
man kiss her", treating "man" as the object "saw" -- the slightly more
complex version of the rules in VRULES will avoid this trap, since it
will spot the presence of the verb after "man"). Note that it handles
the potential for adjectives and nouns to be confused by the tagger,
since either will be acceptable as noun modifiers.
"""

TAGS = """
verb = (VV|VH)
det = (DT|AT|CR|DP)
nmod = (AJ|NN)
noun = NN
to = TO
adv = AV
aux = (VH|VB|VM)
"""

VRULES = """
verb det? nmod* noun verb
|
(?P<TVERB>verb) det? nmod* (?P<OBJ>noun)
"""

OBJRULES = """
(?P<TVERB>verb) det? nmod* (?P<OBJ>noun)
"""

SUBJRULES = """
det? nmod* (?P<SUBJ>noun) aux* adv* (?P<VERB>verb)
"""

def makeTagTable(tags=TAGS):
    tagtable = {}
    for x in TAGS.strip().split("\n"):
        x = x.split("=")
        tagtable[x[0].strip()] = x[1].strip()
    return tagtable

def expandPatterns(rule, tagtable):
    if isinstance(tagtable, str):
        tagtable = makeTagTable(tagtable)
    for tag in tagtable:
        rule = rule.replace(tag, "\s*(\S*:%s)"%(tagtable[tag]))
    return re.compile(re.compile("\s").sub("", rule))

def add2pairs(f0, g, f1, pairs):
    if not g in pairs:
        pairs[g] = {}
    if not f0 in pairs[g]:
        pairs[g][f0] = {}
    if f1 in pairs[g][f0]:
        pairs[g][f0][f1] += 1
    else:
        pairs[g][f0][f1] = 1
        
def parseSentence(text, pattern, tagtable=None, pairs=None, root=root):
    az = re.compile("^[a-zA-Z]*$")
    """
    pairs is a table of the form 

    {'OBJ': {'acquire': {'deficiency': 1}, 'affect': {'body': 1}, 'fight': {'infection': 1}}, 
     'TVERB': {'deficiency': {'acquire': 1}, 'body': {'affect': 1}, 'infection': {'fight': 1}}}

    i.e. 'OBJ' is a table of verbs and their objects and 'TVERB' is a
    table of nouns and the verbs they can be objects of
    """
    if pairs is None: pairs = {}
    """
    Make sure that we have actually compiled the pattern
    """
    if isinstance(pattern, str):
        pattern = expandPatterns(pattern, tagtable)
    for i in pattern.finditer(text):
        l = []
        """
        g0 and g1 are the names of named groups in the rule, i.e. of what we are looking for

        form0 and form1 are the items that are associated with those group names

        Add them to the table as {g0: {form0: {form1: count1}}} and
        {g1: {form1: {form0: count0}}}, i.e. record a verb and its
        object both ways round.
        """
        for g0 in pattern.groupindex:
            if g0.isupper() and i.group(g0):
                form0 = root(i.group(g0).split(":")[0].strip().lower())
                for g1 in pattern.groupindex:
                    if g1 > g0 and g1.isupper() and i.group(g1):
                        form1 = root(i.group(g1).split(":")[0].strip().lower())
                        if az.match(form0) and az.match(form1):
                            add2pairs(form1, g0, form0, pairs)
                            add2pairs(form0, g1, form1, pairs)
    return pairs

def regexParse(pattern, tagged, tagtable=None, pairs=None):
    if tagtable is None:
        tagtable = makeTagTable()
    if isinstance(pattern, str):
        pattern = expandPatterns(pattern, tagtable)
    if pairs is None: pairs = {}
    for s in tagged:
        """
        s is the output from a tagger, so is expected to be of the form

        [('AIDS', 'NN'), ('(', 'PU'), ('Acquired', 'VV'), ('Immune', 'AJ'), ('Deficiency', 'NN'), ...]

        To match it with our pattern we need it to look like

        AIDS:NN (:PU Acquired:VV Immune:AJ Deficiency:NN
        """
        s = " ".join("%s:%s"%(x[0], x[1]) for x in s)
        parseSentence(s, pattern, pairs=pairs)
    return pairs
