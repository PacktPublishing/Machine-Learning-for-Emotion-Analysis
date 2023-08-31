
"""
There are a million taggers out there. This one is easy and fast to
train, fairly accurate (more accurate than the NLTK standard tagger,
though this does depend on the size and quality of the training data),
and fairly fast (about 4 times faster than the NLTK standard tagger).

Once you've trained a tagger, which you do by doing

>>> tagger = TAGGER(<taggedcorpusreader>)))
 
where <taggedcorpusreader> is something which produces tagged text,
you can call it by

>>> tagger("she was the great love of his life", tokeniser=tokenise)

or

>>> tagger("she was the great love of his life", tokeniser=tokenise)

The first of these just returns the commonest tag for each word in the
training data: this is surprisingly accurate (well, it's bound to be
right most of the time!) and is very fast (280K words/second) but it
must get some words wrong sometimes.

t.tag("she was the great love of his life", tokeniser=tokenise)

she:PNP was:VBD the:AT0 great:AJ0 love:NN1 of:PRF his:DPS life:NN1

t.tag("I love her", tokeniser=tokenise)

I:PNP love:NN1 her:DPS

The commonest use of "love" in the BNC is as a noun, so it gets
labelled as such in both sentences.

So you need to take the local context into account. There are numerous
ways of doing this, with potential trade-offs between time taken to
train, time taken to apply and accuracy. The code below implements an
HMM-based approach -- fast to train, fairly fast to execute (730K
words/second), reasonably accurate (depends on the nature of the
training set).

t("she was the great love of his life", tokeniser=tokenise)
she:PNP was:VBD the:AT0 great:AJ0 love:NN1 of:PRF his:DPS life:NN1

t("I love her", tokeniser=tokenise)
I:PNP love:VVB her:PNP
"""

import re
from basics.utilities import *
from corpora import *


def lookup(x, table, default):
    if x in table:
        return table[x]
    else:
        return table[default]

def addToTags(word, tag, tags):
    if not word in tags:
        tags[word] = {}
    if tag in tags[word]:
        tags[word][tag] += 1
    else:
        tags[word][tag] = 1

def getBest(d):
    t = False
    best = False
    for x in d:
        if not t or d[x] > t:
            t = d[x]
            best = x
    return best, t

def isnumber(n):
    try:
        float(n.replace(",", ""))
        return True
    except:
        return False

"""
Very simple tagger: just return the commonest tag for the word. Two small complications:

(i) we use suffixes and prefixes in order to assign a tag to an
unknown word, e.g. we would like the word "bedabbled" to be classified
as a verb because most words that end with "-ed" are verbs. In some
languages the first few letters tend to be significant, in some the
last few do, and in some both do. So when training your tagger you
should specify how many letters should be used at the start
(usePrefix) and end (usePrefix). The default settings are usePrefix=0,
useSuffix=4, since these work nicely for English.

(ii) BASETAGGERs just use the word frequency (and prefix/suffix
frequency) tables. We are going to want to make an HMMTAGGER as a
specialisation of this, so we collect the transition tables as we go
even when just making a BASETAGGER.
"""
class BASETAGGER:
    
    def __init__(self, corpus, usePrefix=0, useSuffix=4, exceptions={}, default=sys.maxsize, taglength=2):
        self.tags = {}
        self.usePrefix = usePrefix
        self.useSuffix = useSuffix
        self.transitions = {}
        self.default = default
        prev = False
        try:
            N = len(corpus)
        except:
            N = None
        T0 = time.time()
        for i, (word, tag) in enumerate(corpus):
            progress(i, N, T0)
            tag = tag[:taglength]
            if word.lower() in exceptions:
                tag = exceptions[word.lower()]
            elif isnumber(word):
                tag = "CR"
            else:
                tag = tag.split("-")[0]
            if prev:
                if not prev in self.transitions:
                    self.transitions[prev] = {}
                if tag in self.transitions[prev]:
                    self.transitions[prev][tag] += 1
                else:
                    self.transitions[prev][tag] = 1
            prev = tag
            addToTags(word, tag, self.tags)
            if self.default:
                addToTags(self.default, tag, self.tags)
            for p in range(usePrefix):
                addToTags("%s-"%(word[:p]), tag, self.tags)
            for s in range(useSuffix):
                addToTags("-%s"%(word[-s:]), tag, self.tags)
        for word in self.tags:
            softmax(self.tags[word])
        for tag in self.transitions:
            softmax(self.transitions[tag])

    def lookup(self, word):
        if word in self.tags:
            return self.tags[word]
        prefixtags = False
        for i in range(self.usePrefix, 0, -1):
            prefix = "%s-"%(word[:i+1])
            print(prefix)
            if prefix in self.tags:
                prefixtags = self.tags[prefix]
                break
        suffixtags = False
        for i in range(self.useSuffix, 0, -1):
            suffix = "-%s"%(word[-i:])
            if suffix in self.tags:
                suffixtags = self.tags[suffix]
                break
        if prefixtags:
            if suffixtags:
                tags = {tag: prefixtags[tag]*suffixtags[tag] for tag in prefixtags if tag in suffixtags}
                if tags:
                    return normalise(tags)
                else:
                    return self.default
            else:
                return prefixtags
        elif suffixtags:
            return suffixtags
        else:
            return self.default

    def basetag(self, text, tokeniser=tokenise, pretty=False):
        tags = []
        for word in tokeniser(text):
            tag = self.lookup(word)
            tags.append((word, sortTable(tag)[0][0]))
        if pretty:
            return " ".join("%s:%s"%(k) for k in tags)
        else:
            return tags

    def __call__(self, text, tokeniser=tokenise, pretty=False):
        return self.basetag(text, tokeniser=tokeniser, pretty=pretty)

class HMMTAGGER(BASETAGGER):
    
    def retraceLinks(self):
        last = sortTable(self.nodes[-1])[0][0]
        tags = [last]
        for ltable in reversed(self.links):
            last = ltable[last]
            tags.append(last)
        return reversed(tags)
        
    """
    p(T_i) = e(T_i)*argmax(p(T_{i-1})*f(T_{i-1}->t{i}))
    """
    
    def fulltag(self, text, tokeniser=tokenise, pretty=False):
        text = tokenise(text)
        nodes = [{"CR": 1} if isnumber(w) else softmax(self.lookup(w), copy=True) for w in text]
        links = []
        last = nodes[0]
        for word, current in zip(text[1:], nodes[1:]):
            link = {}
            links.append(link)
            for tag in current:
                link[tag] = {}
                m = 0
                l = None
                for prev in last:
                    if prev in self.transitions and tag in self.transitions[prev]:
                        t = self.transitions[prev][tag]*last[prev]*current[tag]
                        if t > m:
                            m = t
                            l = prev
                if l is None:
                    prev = getBest(last)[0]
                    link[tag] = prev
                    current[tag] = current[tag]*last[prev]
                else:
                    current[tag] = m
                    link[tag] = l
            last = softmax(current, False)
        self.nodes = nodes
        self.links = links
        tags = list(zip(text, self.retraceLinks()))
        if pretty:
            return " ".join("%s:%s"%(k) for k in tags)
        else:
            return tags

    def __call__(self, text, tokeniser=tokenise, pretty=False, base=False):
        if base:
            return self.basetag(text, tokeniser=tokeniser, pretty=pretty)
        else:
            return self.fulltag(text, tokeniser=tokeniser, pretty=pretty)

        
from nltk.tag import pos_tag

class NLTKTAGGER():

    def __init__(self):
        self.tagger = pos_tag

    def __call__(self, s, pretty=False):
        tags = self.tagger(tokenise(s, tokenisePattern=ENGLISHPATTERN))
        if pretty:
            return " ".join("%s:%s"%(k) for k in tags)
        else:
            return tags
    
"""
In order to compare taggers you obviously have to ensure that they use
the same tagsets. This sounds easy enough, e.g. the BNC uses PR as the
tag for prepositions whereas the UDT convention uses IN, so why not
just map PR to IN?

There are a number of problems with this:

(i) one tagger may assign more fine-grained tags than another -- the
NLTK tagger says that "have", "do" and "be" are all just verbs ("VB"),
the HMM-based one trained on the BNC assigns them the labels "VH",
"VD" and "VB". The NLTK taggers says the "his" and "our" are pronouns,
the HMM-based one says that they are possessive pronouns. It is
unreasonable to say that the NLTK tagger is wrong in these cases, but
mapping "VH" to "VB", or mapping "DP" to "PR", loses information. In
order to make scoring easier, it can be helpful to map the more
detailed tags to the less detailed ones, but if the more detailed ones
are actually more useful for subsequent stages of processing then the
tagger that provides them should be preferred.
 
(ii) different linguistic theories treat closed class words
differently. The NLTK tagger says that "as" is a preposition, the
HMMTAGGET trained on the BNC says it's a coordinating conjunction, the
HMMTAGGER trained on the UDT English datasets treats as either an
adverb or a conjunction, depending on the context (I:PRON loved:VERB
her:PRON as:ADV soon:ADV as:SCONJ I:PRON met:VERB her:PRON just:ADV
as:SCONJ she:PRON loved:VERB me:PRON as:ADV though:SCONJ I:PRON
were:AUX a:DET nice:ADJ person:NOUN); the NLTK tagger says that in
sentences like "Official figures suggest that ACET provided care ..."
the word "that" is a preposition, the HMM tagger says it's a
coordinating conjunction. Which of these you believe is right depends
on your linguistic theory. The NLTK tagger says that "that" in this
sentence is a preposition because that's what the guidelines that the
annotators for the training data were told to say, the HMM-BNC one
says that it's coordinating conjunction because that's what the
annotators for its training data were told to say. The only reasonable
way to judge such cases is to say whether the tagger got a particular
case right *on its own terms*

(iii) similarly, if the data that a tagger was trained on includes a
lot of words in all upper case, or a lot of words that begin with
upper case letters but aren't proper names, then it will do better on
test data that contains words in all upper case or words that begin
with upper case letters but aren't proper names than one that was
trained on data without such examples. The HMM-based tagger performs
*much* better on such data than the NLTK tagger, but since the NLTK
tagger was not aimed at cases like this it is unfair to compare them
on it.

The scores reported in the book take this into account. These scores
were based on looking at cases where the two taggers assigned
different tags to open-class words written entirely in lower case. It
is not possible to compare them on closed class words, since the
choice of what tag to assign to a closed-class word is driven by the
instructions given to the annnotators, which is in turn driven by the
underlying linguistic theory, so as long as the assignments are
consistent they have to be treated as being right; and it is unfair to
compare the two taggers on non-lower-case words when it is clear that
the NLTK tagger was not trained on such data and hence performs very
poorly on it.
"""

def applyAllTaggers(s, alltaggers, zipped=False):
    zipped = [[""]+s.split()]
    for t in alltaggers:
        zipped.append([t.__class__.__name__]+[tag for word, tag in t(s)])
    return ["\t".join(r) for r in list(zip(*zipped))]
