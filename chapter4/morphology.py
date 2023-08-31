from utilities import *
from nltk.corpus import wordnet
from corpora import *

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

def fixaffixes(affixes):
    p = re.compile("\s*;\s*")
    return {a: [readTerms(s)[0] for s in p.split(affixes[a].strip())] for a in affixes}

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

def readAllWords(nltkdata="/Users/ramsay/nltk_data/corpora/wordnet", base=False):
    allwords = collection()
    if base:
        affixes = {"v": "v->tense",
                   "n": "n->num",
                   "a": "a->cmp",
                   "r": "r"}
    else:
        affixes = {"v": "v[tense=T, finite=F, number=N, person=P]->tense[tense=T, finite=F, number=N, person=P]",
                   "n": "n[number=N]->num[number=N]",
                   "a": "a[comp=C]->cmp[comp=C]",
                   "r": "r"}
        
    p = re.compile("(?P<form>[a-z]{3,}) (?P<pos>n|v|r|a) ")
    for f in os.listdir(nltkdata):
        if f.startswith("index."):
            for i in p.finditer(open(os.path.join(nltkdata, f)).read()):
                form = i.group("form")
                if form:
                    terms = readTerms(affixes[i.group("pos")])[0]
                    if not form in allwords or not terms in allwords[form]:
                        allwords.add(form, terms)
    return allwords

try:
    ALLWORDS
except:
    ALLWORDS = readAllWords(base=False)
    BASEWORDS = readAllWords(base=True)

"""
The rules: we need to deal with "came" as a rule rather than a special
case because we have to allow for became and overcame, and likewise we
want to allow for "earnt" and "learnt".
"""

SPELLINGRULES = """
ee X:(d|n) ==> ee + e X
C y X:ing ==> C ie + X
C X:ly ==> C le + X
C aid ==> C ay + ed
i X1:(e(d|r|st)|ly|ness) ==> y + X1
i e s ==> y + s
X:((d|g|t)o)|x|s(h|s)|ch es ==> X + s
C0 rous ==> C0 e r + ous
C0 i C1 (?!(.*(a|e|i|o|u))) ==> C0 y + C1
X0 (?!(?P=X0)) C X1:e(d|n|r|st)|ing (?!(.*(e|i))) ==> X0 C e + X1
^ C V0 X1:e(d|n|r|st)|ing (?!(.*(e|i))) ==> C V0 e + X1
u X1:e(d|n|r|st)|ing ==> ue + X1
C0 V C1 C1 X:(e(d|n|r|st)|ing) ==> C0 V C1 + X
"""

BASEPREFIXES = fixaffixes(
    {"un": "(v->tense)->(v->tense); (a->cmp)->(a->cmp)",
     "re": "(v->tense)->(v->tense)",
     "dis": "(v->tense)->(v->tense)"})

PREFIXES = fixaffixes(
    {# "un": "(v[]->tense[])->(v[]->tense[]); (a[]->cmp[])->(a[]->cmp[])",
     # "re": "(v[]->tense[])->(v[]->tense[])",
     # "dis": "(v[]->tense[])->(v[]->tense[])",
     })

BASESUFFIXES = fixaffixes(
    {
        "": "tense; num; cmp",
        "ing": "tense",
        "ed": "tense",
        "s": "tense; num",
        "en": "tense",
        "est": "cmp",
        "ly": "r<-a; r<-v",
        "ic": "a<-(n->num)",
        "al": "a<-a",
        "er": "(n->num)<-v; cmp",
        "ment": "((n->num)<-(v->tense))",
        "ous": "a<-(n->num)",
        "less": "a<-(n->num)",
        "ness": "(n->num)<-(v->tense)",
        "able": "a<-(v->tense)",
     })

SUFFIXES = fixaffixes(
    {"": "tense[finite=infinitive]; tense[finite=tensed, tense=present]; num[number=singular]; cmp[comp=base]",
     "ing": "tense[finite=participle, tense=present]",
     "ed": "tense[finite=participle, tense=present, voice=passive]; tense[tense=past, voice=active]",
     "s": "tense[finite=tensed, tense=present, number=singular, person=third]; num[number=plural]",
     "en": "tense[finite=participle]",
     "est": "cmp[comp=superlative]",
     "ly": "r[]<-a[comp=base]; r[]<-v[finite=participle, tense=present]",
     "ic": "a[comp=base]<-(n->num)",
     "al": "a[comp=base]<-a[comp=base]",
     "er": "(n[]->num[])<-v[finite=infinitive]; cmp[comp=comparative]",
     "ment": "(n[]->num[])<-v[finite=infinitive]",
     "ous": "a[comp=base]<-(n[]->num[])",
     # "less": "a[comp=base]<-(n[]->num[])",
     "ness": "(n[]->num[])<-a[comp=base]",
     "able": "a[comp=base]<-(v[]->tense[])",})

FSPELLING = """
ive ==> if+e
"""

"""
FPREFIX = fixaffixes({})

FSUFFIXES = fixaffixes({
    "": "gen; num", "s": "num", "e": "gen; person",
    "er": "mood", "": "mood",
    "ez": "person", "ais": "person", "a": "person", "ai": "person", "aient": "person",
    "ait": "person", "as": "person",  "asse": "person", "asses": "person", "ent": "person",
    "es": "person", "iez": "person", "ions": "person", "ons": "person", "ont": "person", "ât": "person",
    })

FWORDS = {"noir": "(a->num)->gen",
          "sportif": "(a->num)->gen",
          "regard": "(v->person)->mood",}

for f in FWORDS:
    FWORDS[f] = [readTerms(FWORDS[f])[0]]
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
    pattern = re.compile("(%s).*"%("|".join(p[0] for p in patterns)))
    return pattern, [p[1] for p in patterns]

fsts = compileFSTs(SPELLINGRULES)

def applyFSTs(form, fsts, n=0):
    m = fsts[0].search(form)
    if m:
        k = m.group(0)
        g = m.groupdict()
        for x in g:
            if not g[x] is None:
                k1 = fsts[0].sub(fsts[1][int(x.split("_")[-1])], k)
                return form[:-len(k)]+k1

def deref(v, bindings):
    while v in bindings:
        v = bindings[v]
    return v

def applyBindings(d, bindings):
    if isinstance(d, dict):
        return {k: deref(d[k], bindings) for k in d}
    else:
        return d

def unify(x, y, bindings=None):
    if bindings is None:
        bindings = {}
    if x == y:
        return bindings
    elif isinstance(x, dict) and isinstance(y, dict):
        for k in x:
            vx = deref(x[k], bindings)
            if k in y:
                vy = deref(y[k], bindings)
                if vx == vy:
                    pass
                else:
                    if isinstance(vx, str) and vx.isupper():
                        bindings[vx] = vy
                    elif isinstance(vy, str) and vy.isupper():
                        bindings[vy] = vx
                    else:
                        return False
            else:
                pass
        return bindings
    else:
        if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
            for a, b in zip(x, y):
                bindings = unify(a, b, bindings)
                if bindings is False:
                    return False
            return bindings
        return False
                    
def showterm(d, showvars=True):
    if isinstance(d, dict):
        return "[%s]"%(", ".join("%s=%s"%(k, d[k]) for k in d if showvars or not d[k].isupper()))
    elif isinstance(d, tuple):
        return "".join(showterm(x, showvars=showvars) for x in d)
    else:
        return d

def combine(l0, l1, n=0):
    l = []
    for x0 in l0:
        for x1 in l1:
            # print("%sX0 %s\n%sX1 %s"%(" "*n, x0, " "*n, x1))
            if isinstance(x0, tuple) and x0[1] == "->":
                u = unify(x0[2], x1)
                if not u is False:
                    # print("%sCOMBINED %s %s -> %s"%(" "*n, x0, x1, applyBindings(x0[0], u)))
                    l.append(applyBindings(x0[0], u))
            if isinstance(x1, tuple) and x1[1] == "<-":
                u = unify(x1[2], x0)
                if not u is False:
                    # print("%sCOMBINED %s %s -> %s"%(" "*n, x0, x1, applyBindings(x1[0], u)))
                    l.append(applyBindings(x1[0], u))
    # print("%sL0 %s, L1 %s, L %s"%(" "*n, l0, l1, l))
    return l

def checkParts(word, part1, part2, table, prefixes, allwords, suffixes, ALLWORDS, fsts=fsts, n=0):
    if PRINTING: print("%scheckParts(%s, %s, %s)"%(" "*n, word, part1, len(table)))
    if part1 in table:
        if PRINTING: print("%sWORD %s, PART1 %s, PART2 %s, T %s %s"%(" "*n, word, part1, part2, len(table), len(allwords)))
        combined = combine(word[1], table[part1], n)
        if PRINTING: print("%sCOMBINED %s"%(" "*n, combined))
        if table == ALLWORDS:
            part1 = "#%s#"%(part1)
        if combined:
            for k in splitword(("%s+%s"%(word[0], part1), combined), part2, prefixes, allwords, suffixes, ALLWORDS, fsts=fsts, n=n+1):
                yield k
                
def splitword(word, form, prefixes, allwords, suffixes, ALLWORDS, fsts=fsts, n=0):
    # print("%ssplitword: WORD %s, FORM %s"%(" "*n, word, form))
    if form == "":
        yield word
    if word:
        for i in range(len(form)+1):
            part1, part2 = form[:i], form[i:]
            for k in checkParts(word, part1, part2, suffixes, [], [], suffixes, ALLWORDS, fsts, n=n+1):
                yield k
            for k in checkParts(word, part1, part2, allwords, [], [], suffixes, ALLWORDS, fsts, n=n+1):
                yield k
        if fsts:
            m = applyFSTs(form, fsts)
            if m:
                part1, part2 = m.split("+")
                for k in checkParts(word, part1, part2, suffixes, [], [], suffixes, ALLWORDS, fsts, n=n+1):
                    yield k
                for k in checkParts(word, part1, part2, allwords, [], [], suffixes, ALLWORDS, fsts, n=n+1):
                    yield k
    else:
        for i in range(len(form)+1):
            part1, part2 = form[:i], form[i:]
            if part1 in allwords:
                for k in splitword(("#%s#"%(part1), allwords[part1]), part2, [], allwords, suffixes, ALLWORDS, fsts=fsts, n=n+1):
                    yield k
            if part1 in prefixes:
                for k in splitword((part1, prefixes[part1]), part2, prefixes, allwords, suffixes, ALLWORDS, fsts=fsts, n=n+1):
                    yield k
        if fsts:
            m = applyFSTs(form, fsts)
            if m:
                part1, part2 = m.split("+")
                if part1 in allwords:
                    for k in splitword(("#%s#"%(part1), allwords[part1]), part2, [], allwords, suffixes, ALLWORDS, fsts=fsts, n=n+1):
                        yield k
                if part1 in prefixes:
                    for k in splitword((part1, prefixes[part1]), part2, prefixes, allwords, suffixes, ALLWORDS, fsts=fsts, n=n+1):
                        yield k

def chooseForm(forms):
    l = []
    N = -1
    for form in forms:
        n = sum(0.5 if x == "" else 1 for x in form[0].split("+"))
        if n > N:
            N = n
    for form in forms:
        if N == sum(0.5 if x == "" else 1 for x in form[0].split("+")):
            l.append((form[0], showterm(form[1])))
    return l

def splitallwords(words, prefixes, allwords, suffixes, fsts=fsts, out=False, pattern=".*"):
    if isinstance(words, dict):
        words = sorted(words.keys())
    if out:
        if not out == sys.stdout:
            out = open(out, "w")
    pattern = re.compile(pattern)
    l = collection()
    for word in words:
        if AZ.fullmatch(word):
            if not pattern.match(word):
                continue
            k = chooseForm(list(splitword(False, word, prefixes, allwords, suffixes, allwords, fsts=fsts)))
            # print("K %s"%(k))
            try:
                s = k[0][0].split("+")
                l[word] = k
                if out:
                    out.write("%s\t%s\n"%(word, k))
            except:
                pass
    if out and not out == sys.stdout:
        out.close()
    return l

def justroot(word):
    l = splitallwords([word], PREFIXES, ALLWORDS, SUFFIXES, fsts=fsts, out=False, pattern=".*")
    if word in l:
        return re.compile(".*#(?P<root>.*)#.*").match(l[word][0][0]).group("root")
    else:
        return word

def morphyroot(word):
    root = word
    for t in 'nvra':
        x = wordnet.morphy(word, t)
        if x and len(x) < len(root):
            root = x
    return root

def morphyallwords(words):
    return [morphyroot(word) for word in words if AZ.fullmatch(word)]

p = re.compile("(?P<x>..*?)(?P=x)+")

def checktaggers(words, n=sys.maxsize):
    seen = set()
    for word in words:
        r1 = justroot(word)
        r2 = morphyroot(word)
        if not r1 == r2 and not word in seen:
            print(word, r1, r2)
            seen.add(word)
        n -= 1
        if n < 0:
            return
        
PRINTING = False

SPANISHENDINGS = re.compile("(?<=...)(((?<=a|e|o|i|u)s)|((a|i|e)r)|(a|e|i)(mos|n)|(ái|í|éi)s|((a|o)s?)|é|(a|i)ste|i?ó|(a|i)steis|(a|ie)ron|í)(l(a|o|e)s?)?$")

def stemSpanish(word):
    return SPANISHENDINGS.sub("", word)
