from .stem3 import readAllWords

PREFIXES = fixaffixes(
    {"un": "(v[]->tense[])->(v[]->tense[]); (a[]->cmp[])->(a[]->cmp[])",
     "re": "(v[tense=T, finite=F, number=N, person=P]->tense[tense=T, finite=F, number=N, person=P])->(v[]->tense[])",
     "dis": "(v[]->tense[])->(v[]->tense[])",
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

ROOTS = {"v": "v[tense=T, finite=F, number=N, person=P]->tense[tense=T, finite=F, number=N, person=P]",
           "n": "n[number=N]->num[number=N]",
           "a": "a[comp=C]->cmp[comp=C]",
           "r": "r"}

ALLWORDS = readAllWords(roots=ROOTS)

SUFFIXFSTS = re.compile("^(%s)"%("|".join(SUFFIXES)))

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
    # print("UNIFY %s %s"%(x, y))
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
                    print("%sCOMBINED %s %s -> %s"%(" "*n, x0, x1, applyBindings(x1[0], u)))
                    l.append(applyBindings(x1[0], u))
    # print("%sL0 %s, L1 %s, L %s"%(" "*n, l0, l1, l))
    return l
