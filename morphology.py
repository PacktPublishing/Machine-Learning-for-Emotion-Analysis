from utilities import *
from nltk.corpus import wordnet

"""
English morphology would be straightforward if (i) adding affixes
didn't change the spelling of the root -- just remove "ed" from
"walked", "ing" from "walking", ..., and (ii) there weren't numerous
irregular cases -- "thought" = "think"+"ed", "children" = "child"+"s",
...

We start by thinking about the spelling changes. As discussed in the
book, these occur partly because adding an affix can alter the
pronunciation -- it would be difficult to pronounce /b/ /o/ /x/ /s/,
which is what you would get if you just added tha standard plural
ending onto "box". It's much easier to pronounce it if you insert a
vowel (a schwa, or indistinc unstressed vowel) between the /x/ and the
/s/, and the spelling reflects this. More complicatedly, English
spelling includes a number of rules that tell you about pronunciation,
e.g. the use of "magic E" to show that the previous vowel is long --
"cap" vs "cape". These rules can be cancelled or changed when affixes
are added, e.g. the past tense forms of these two become "capped" (wih
the double-p marking the fact that the vowel is short) and "caped".

Disentangling these two is tricky. Fortunately we don't have to think
too much about *why* some spelling change happens, all we need to is
capture the fact that it does. We can do this using regexes, with the
convention that upper case C0, C1, ... will match any consonant, V0,
V1, ... will match any vowel and X0, X1, ... will match any letter.

The rules below say that if the surface string matches the left-hand
side of a rule it *may* be possible to rewrite as two parts, as given
by the right hand side, where variables on the left get copied onto
the right. The third rule, for instance, says that "agreed" is
"agree"+"ed" and "seen" is "see"+"en".

You can't just apply the rules to a surface string, because if you do
you'll get lots of false positive -- you'll read "water" as
"wate"+"er", and "steed" as "stee"+"ed". Treating a word as a root + a
suffix only makes sense if there is actually a suitable root --
"freed" is "free"+"ed" because "free" is a verb, "steed" isn't
"stee"+"ed" because "stee" isn't. So you need a list of roots, which
we will get from the NLTK implementation of WordNet.

But any list of roots will have omissions and oddities. The list we
get from the NLTK includes some pretty strange words, which means that
we get our rules being applied in places where we don't want them, and
we also have all the irregular forms ("brought"="bring"+"ed",
"stolen"="steal"+"en"). To cope with these we just start with a list
of exceptions. The irregular forms are all obvious enough (probably
incomplete, but that's easy to fix). Cases like "master": ("master",
"") are there because "mast" is a word, so it's conceivable that it
has a comparative form "master". We include 'blocking' cases like
these to cover obvious odd analyses.
"""

"""
Get all the standard words (no upper case, no numbers, no-hyphens) in the wordnet list
"""
AZ = re.compile("^[a-z]*$")
ALLWORDS = set(w for w in wordnet.all_lemma_names() if AZ.match(w))

"""
The rules: we need to deal with "came" as a rule rather than a special
case because we have to allow for became and overcame, and likewise we
want to allow for "earnt" and "learnt".
"""

FSTS = """
came ==> come + ed
earnt ==> earn + ed
ee X0:(d|n) ==> ee + e X0
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

"""
Compile a rule like 

   ee X0:(d|n) ==> ee + e X0

to a regex and a substitution like

  (re.compile('ee(?P<X0>(d|n))$'), 'ee+e\\g<X0>')

The pattern will match a sequence like "een" or "eed" at the end of a word, and will bind the group X0 to "n" (or "d")). The substition group will then convert "seen" to "see+en" or "freed" to "free+ed"

"""
def compileFST(fst):
    [lhs, rhs] = fst.split("==>")
    pattern = ""
    groups = set()
    for x in lhs.strip().split():
        """ 
        Upper cases elements are named groups
        """
        if x[0].isupper():
            c, n = x[0], x[1]
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
            elif c == "X" and x[2] == ":":
                range = x[3:]
            else:
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
            pattern += x
    pattern += "$"
    rep = ""
    for r in rhs.strip().split():
        if r[0].isupper():
            rep += "\g<%s>"%(r)
        else:
            rep += r
    return re.compile(pattern), rep

"""
Compile your list of rules into a list of rules. It might be slightly
quicker to compile the whole list of rules into a single disjunctive
pattern, but it makes looking after the bindings more complex and it
makes it hard to backtrack if you found something that matched but
there was no such root.
"""
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
                                  "sought": ("seek", "ed"),
                                  "thought": ("think", "ed"),
                                  "caught": ("catch", "ed"),
                                  "taught": ("teach", "ed"),
                                  "stolen": ("steal", "en"),
                                  "stole": ("steal", "ed"),
                                  "broken": ("break", "en"),
                                  "broke": ("break", "ed"),
                                  "took": ("take", "ed"),
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
                                  "king": ("king", ""),
                                  "drank": ("drink", "ed"),
                                  "drunk": ("drink", "en"),
                                  "cooked": ("cook", "ed"),
                                  "cooker": ("cook", "er"),
                                  "cooking": ("cook", "ing")}):
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

        


 
 
                 



                   
