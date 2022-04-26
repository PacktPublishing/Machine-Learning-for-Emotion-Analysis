
"""
There are a million taggers out there. This one is easy and fast to
train, fairly accurate (more accurate than the NLTK standard tagger,
though this does depend on the size and quality of the training data),
and fairly fast (about 4 times faster than the NLTK standard tagger).

Once you've trained a tagger, which you do by doing

>>> tagger = TAGGER(reader(os.path.join(BNC, "B"), lambda w: BNCTaggedWordReader(w, specials={"to":"TO", "that": "THAT"})))

(e.g. to train it on section A of the BNC by providing a tagged reader
for that set of files)

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
from utilities import *
from corpora import *
import dtw

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
    
class TAGGER:
    
    def __init__(self, corpus, usePrefix=0, useSuffix=4, exceptions={}, default=sys.maxsize):
        self.tags = {}
        self.usePrefix = usePrefix
        self.useSuffix = useSuffix
        self.transitions = {}
        self.default = default
        prev = False
        for word, tag in corpus:
            if word.lower() in exceptions:
                tag = exceptions[word.lower()]
            elif isnumber(word):
                word = "12345"
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

    def tag(self, text, tokeniser=tokenise):
        tags = []
        for word in tokeniser(text):
            tag = lookup(word, self.tags, self.default)
            tags.append((word, sortTable(tag)[0][1]))
        return list(tags)

    def retraceLinks(self):
        last = sortTable(self.nodes[-1])[0][1]
        tags = [last]
        for ltable in reversed(self.links):
            last = ltable[last]
            tags.append(last)
        return reversed(tags)
        
    """
    p(T_i) = e(T_i)*argmax(p(T_{i-1})*f(T_{i-1}->t{i}))
    """
    
    def __call__(self, text, tokeniser=tokenise):
        text = tokenise(text)
        if text == []:
            return zip([], [])
        nodes = [{"CR": 1} if isnumber(w) else softmax(lookup(w, self.tags, self.default), copy=True) for w in text]
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
        return list(zip(text, self.retraceLinks()))

"""
In order to compare taggers you obviously have to ensure that they use
the same tagsets. This sounds easy enough, e.g. the BNC uses PR as the
tag for prepositions whereas the UDT convention uses IN, so why not
just map PR to IN?

There are a number of problems with this:

(i) one tagger may assign more fine-grained tags than another -- the
NLTK tagger says that "have", "do" and "be" are all just verbs ("VB"), the
HMM-based one assigns them the labels "VH", "VD" and "VB". The NLTK
taggers says the "his" and "our" are pronouns, the HMM-based one says
that they are possessive pronouns. It is unreasonable to say that the
NLTK tagger is wrong in these cases, but mapping "VH" to "VB", or
mapping "DP" to "PR", loses information. In order to make scoring
easier, it can be helpful to map the more detailed tags to the less
detailed ones, but if the more detailed ones are actually more useful
for subsequent stages of processing then the tagger that provides them
should be preferred.

(ii) different linguistic theories treat closed class words
differently. The NLTK tagger says that "as" is a preposition, the HMM
tagger says it's a coordinating conjunction; the NLTK tagger says that
in sentences like "Official figures suggest that ACET provided care
..." the word "that" is a preposition, the HMM tagger says it's a
coordinating conjunction. Which of these you believe is right depends
on your linguistic theory. The NLTK tagger says that "that" in this
sentence is a preposition because that's what the guidelines that the
annotators for the training data were told to say, the HMM-based one
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

def BNC2UDT(tag):
    try:
        """
        table for rewriting BNC tags as UDT for comparison
        """
        return {"NP": "NN",
                "VH": "VB",
                "VD": "VB",
                "VV": "VB",
                "VM": "MD",
                "AJ": "JJ",
                "PN": "PR",
                "PR": "IN",
                "AT": "DT",
                "UN": "POS",
                "XX": "RB",
                "AV": "RB",
                "CJ": "CC",
                "CR": "CD",
                "INTERJ": "UH"}[tag]
    except:
        return tag
    
from nltk.tag import pos_tag

def nltktagger(s):
    return pos_tag(tokenise(s, tokenisePattern=ENGLISHPATTERN))

"""
compareTaggers compares the output of two taggers. This can be used
for comparing two taggers or for comparing a tagger and a Gold
Standard, since the Gold Standard can easily be viewed as a generator
of (form, tag) pairs (see reader2goldstandard below).

Before we can do anything we have to cope with the fact that
tokenising the input for one tagger may involving splitting
differently from how it is split by the other.  We use "dynamic time
warping" to align the outputs of the two taggers so that sections
where they actually split the text differently (e.g. one may split a
hyphenated term like "man-hole" into "man", "-" and "hole" where the
other treats it as a unit, one may split £1.00 into "£" and "1.00" and
the other may treat it as a single item). 

We also need to allow for special words which we want to assign
specific tags (which may differ from those used in the
training/testing corpora, e.g. the decision to just tag the word "to"
with its own unique label).

>>> specials={"to":"TO", "that": "THAT"}

Make a tagger using the data in BNC/TRAINING
>>> tagger = TAGGER(reader(os.path.join(BNC, "TRAINING"), lambda w: BNCTaggedWordReader(w, specials=specials)))

Apply this tagger to the data in BNC/TESTING
>>> tagged = (tagger(sentence) for sentence in reader(os.path.join(BNC, "TESTING"), BNCSentenceReader))

(or tagged = list(tagger(sentence) for sentence in reader(os.path.join(BNC, "TESTING"), BNCSentenceReader)) 
if you want to go over the output more than once, i.e. solidify the generatoy into a fixed list)

And also convert the data in BNC/TESTING into a Gold Standard test set
>>> goldstandard = reader2goldstandard(os.path.join(BNC, "TESTING"), specials={"to":"TO", "that": "THAT"})

And now compare them
>>> score, confusion = compareTaggers(tagged, goldstandard, N=500)

Is the tagger any good? You should always look at the *kinds* of
mistake that a tagger makes as well as the *number* of mistakes it
makes before deciding whether to use it. Some mistakes don't really
matter -- e.g. most of the time it doesn't matter whether something is
tagged as an adjective or a noun, because nouns can function perfectly
well as noun modifiers. It doesn't matter whether "expert" is tagged a
noun or an adjective in

    While numbers of new AIDS cases reported officially each month
    have remained relatively steady , there has been a big increase in
    those needing expert medical and nursing advice at home with a
    24-hour on call back up .

because the next stage of processing (e.g. the regex-based parser in
regexparser.py) will almost certainly allow nouns to be used as
noun-modifiers, so "expert medical advice" will be treated as medical
advice provided by experts no matter whether "expert" is tagged as a
noun or as an adjective. So if all the mistakes that your tagger made
were about labelling things that are to be treated as noun modifiers
as nouns when they should be adjectives then it would actually be
performing perfectly as far as the next stage of processing is
concerned.

compareTaggers returns a confusion matrix as well as the raw counts to
help you look at this.

>>> score, confusion = compareTaggers(tagged, goldstandard, N=5000)

The score gives you an overall percentage of places where the output
of tagger matches the Gold Standard:

>>> print(score)
Total words 5019, tagger agrees with Gold Standard on 4815: percentage agreement 0.959

What are the 4.1% cases that the tagger gets wrong like? Do they
matter? Is there anything we can do about them?

The first tool for investigating this is the confusion matrix:

>>> showConfusion(confusion)
AJ [(298, 'AJ'), (17, 'NN'), (16, 'VV'), (1, 'PR'), (1, 'AV')]
AT [(324, 'AT')]
AV [(222, 'AV'), (4, 'NN'), (3, 'PR'), (3, 'CJ'), (3, 'AJ'), (2, 'DT'), (1, 'VV')]
CJ [(194, 'CJ'), (7, 'PR'), (2, 'AV'), (1, 'VV'), (1, 'PU')]
CR [(120, 'CR'), (2, 'NN'), (1, 'PU'), (1, 'PN')]
DP [(47, 'DP')]
DT [(102, 'DT'), (8, 'AV'), (3, 'AJ')]
EX [(9, 'EX')]
IT [(1, 'IT')]
NN [(1103, 'NN'), (17, 'NP'), (8, 'VV'), (3, 'PU'), (3, 'CR'), (3, 'AV')]
NP [(299, 'NP'), (8, 'NN'), (2, 'UN'), (1, 'PU'), (1, 'CR'), (1, 'AT')]
OR [(14, 'OR'), (4, 'UN'), (1, 'NN')]
PN [(133, 'PN'), (2, 'CR')]
PO [(20, 'PO'), (2, 'VB')]
PR [(521, 'PR'), (3, 'AV'), (2, 'CJ')]
PU [(547, 'PU')]
THAT [(18, 'THAT')]
TO [(135, 'TO')]
UN [(11, 'NN'), (5, 'CR'), (2, 'NP'), (1, 'PU'), (1, 'PR'), (1, 'AJ')]
VB [(166, 'VB'), (1, 'NN')]
VD [(20, 'VD')]
VH [(54, 'VH')]
VM [(53, 'VM'), (1, 'VV')]
VV [(406, 'VV'), (21, 'NN'), (7, 'AJ')]
XX [(20, 'XX')]
ZZ [(4, 'ZZ')]

We see that the three biggest errors are labelling a verb as a noun
(VV [(406, 'VV'), (21, 'NN'), ...]), labelling an adjective as a noun
(AJ [(298, 'AJ'), (17, 'NN'), ...]) and labelling a noun as a proper
name (NN [(1103, 'NN'), (17, 'NP'), ...]). As we've just seen, it may
be that these errors don't really matter, but it's also worth checking
to see if they really are errors. The Gold Standard will generally
have been constructed by human annotators, and *human annotators can
also make mistakes*.

We can use the confusion matrix to see what words have been given a
specific tag in the Gold Standard and a different one by the tagger,
e.g. which words are tagged as verbs in the Gold Standard and as nouns
by the tagger:

>>> confusion["VV"]["NN"]
['JUMBLE', 'work', 'Shopping', 'Daysitting', 'care', 'Nurse', 'care', 'pedal', 'raising', 'experience', 'support', 'Volunteer', 'spending', 'volunteer', 'Issue', 'caring', 'help', 'respite', 'stay', 'planning', 'NOTICE']

findInstances will then find all the instances of a word that has been
given one tag by the tagger and another in the Gold Standard, e.g.

>>> findInstances("nursing", "NN", "VV", tagged, goldstandard)

Catherine qualified in general nursing at Dr Steeven 's Hospital , Dublin .
Catherine:NP qualified:VV in:PR general:AJ nursing:NN at:PR Dr:NP Steeven:NP 's:PO Hospital:NN ,:PU Dublin:NP .:PU
Catherine:NP qualified:VV in:PR general:AJ nursing:VV at:PR Dr:NP Steeven:NP 's:PO Hospital:NN ,:PU Dublin:NP .:PU

We are the largest independent provider of professional home care in the capital giving pain control , nursing and medical advice , 24 hour on call , emotional support and practical volunteer help , including nightsitting .
We:PN are:VB the:AT largest:AJ independent:AJ provider:NN of:PR professional:AJ home:NN care:NN in:PR the:AT capital:NN giving:VV pain:NN control:NN ,:PU nursing:NN and:CJ medical:AJ advice:NN ,:PU 24:CR hour:NN on:PR call:NN ,:PU emotional:AJ support:NN and:CJ practical:AJ volunteer:NN help:NN ,:PU including:PR nightsitting:VV .:PU
We:PN are:VB the:AT largest:AJ independent:AJ provider:NN of:PR professional:AJ home:NN care:NN in:PR the:AT capital:NN giving:VV pain:NN control:NN ,:PU nursing:VV and:CJ medical:AJ advice:NN ,:PU 24:CR hour:NN on:PR call:NN ,:PU emotional:AJ support:NN and:CJ practical:AJ volunteer:NN help:NN ,:PU including:PR nightsitting:VV .:PU

...

In these first two instances of the tagger saying "nursing" is a noun
and the Gold Standard saying it's a verb, the tagger is actually
right: "general nursing" is an NP, and "nursing" is its head, "pain
control , nursing and medical advice" is an NP listing three things --
"pain control" is an NP, "medical advice" is an NP, of "nursing" is
part of a list including these two then it must be an NP. Overall, of
the 19 instances of "nursing" being labelled as a noun by the tagger
and a verb in the Gold Standard, the correct label is noun in 10, verb
in 8 and one is unclear. Likewise, of the 7 instances of "pedal" that
are labelled as verbs in the Gold Standard and as nouns by the tagger,
6 are in fact nouns and 1 is a verb.

*This is inevitable* -- any very large manually labelled corpus will
contain errors, because people make errors, and any very large
automatically labelled corpus will make errors, because algorithms
make errors. So before you decide to use the scores as a way of
choosing between taggers, have a good look at the Gold Standard,
because if it contains lots of errors of a kind that concern you then
you should be very wary of relying too much on the score (this also
applies to places where the tagger and the Gold Standard do agree --
if all the places where the Gold Standard said that "nursing" was a
verb were wrong, then the tagger would also be wrong when it agreed
with it).

The moral? If you want to know whether you should use a given tagger,
you have to look carefully at its "mistakes". They may not matter, and
they may not actually be mistakes. There isn't anything to go on apart
from the scores and the confusion matrix, but you should treat these
very carefully.
"""

def reader2goldstandard(corpus=os.path.join(BNC, "TESTING"), specials={"to":"TO", "that": "THAT"}):
    return reader(corpus,
                  lambda data: BNCTaggedSentenceReader(data, wordreader=lambda data: BNCTaggedWordReader(data, specials=specials)))
    
def compareTaggers(tagged1, tagged2, target=None, N=100):
    k = n = 0
    confusion = {}
    for s1, s2 in zip(tagged1, tagged2):
        forms = " ".join([x[0] for x in s1])
        print("""

%s"""%(forms))
        s = ""
        """
        tagged1 and tagged2 may have been tokenised differently. Use
        dynamic time warping to get the cases where they do actually
        contain the same items, ignore the rest.
        """
        for t1t2 in dtw.ARRAY(s1, s2, EXCHANGE=lambda x1, x2: 0 if x1[0] == x2[0] else 3).align():
            t1 = t1t2[0]
            t2 = t1t2[1]
            if not(t1 == "*" or t2 == "*") and t1[0] == t2[0]:
                form = t1[0]
                tag1 = t1[1]
                tag2 = t2[1]
                if not tag2 in confusion:
                    confusion[tag2] = {}
                if tag1 in confusion[tag2]:
                    confusion[tag2][tag1].append(form)
                else:
                    confusion[tag2][tag1] = [form]
                n += 1
                if t1 == t2:
                    k += 1
                    tags = "%s:%s"%(t1[1], t2[1])
                else:
                    tags = "*%s:%s*"%(t1[1], t2[1])
                s += "%s:%s "%(form, tags)
        print(s)
        score = ("""Total words %s, tagger agrees with Gold Standard on %s: percentage agreement %.3f"""%(n, k, float(k)/n))
        print(score)
        if n >= N:
            break
    return score, confusion

"""
showConfusion just lays out the confusion matrix reasonably neatly.
If latex is set to True, then the output is appropriate latex source
code.
"""
def showConfusion(c):
    for k in sorted(c.keys()):
        t = {x: len(c[k][x]) for x in c[k]}
        print(k, sortTable(t))

def findInstances(form, tag1, tag2, tagged, goldstandard):
    for x1, x2 in zip(tagged, goldstandard):
        if (form, tag1) in x1 and (form, tag2) in x2:
            print("")
            print(" ".join("%s"%(w1[0]) for w1 in x1))
            print(" ".join("%s:%s"%w1 for w1 in x1))
            print(" ".join("%s:%s"%w2 for w2 in x2))
