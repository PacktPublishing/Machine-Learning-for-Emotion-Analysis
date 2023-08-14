PREFIXES = {"", "un", "dis", "re"}

SUFFIXES = {"", "ing", "s", "ed", "en", "er", "est", "ly"}

def stem1(form, PREFIXES, WORDS, SUFFIXES):
    for i in range(len(form)):
        if form[:i] in PREFIXES:
            for j in range(i+1, len(form)+1):
                if form[i:j] in WORDS:
                    if form[j:] in SUFFIXES:
                        yield "%s-%s+%s"%(form[:i], form[i:j], form[j:])


FPREFIXES = {"":"n->n, v->v, a->a"}

FSUFFIXES = {"":"gender, tns, num", "e": "gender", "s":"tns, num"}

def getSuffixes(remainder, SUFFIXES, d=0):
    if d > 2:
        return
    print("R %s"%(remainder))
    if remainder in SUFFIXES:
        yield remainder
    for k in range(0, len(remainder)):
        if remainder[:k] in SUFFIXES:
            for x in getSuffixes(remainder[k:], SUFFIXES, d=d+1):
                yield "%s+%s"%(remainder[:k], x)
                
def stem2(form, PREFIXES, WORDS, SUFFIXES):
    for i in range(len(form)):
        if form[:i] in PREFIXES:
            for j in range(i+1, len(form)+1):
                if form[i:j] in WORDS:
                    for s in getSuffixes(form[j:], SUFFIXES):
                        yield "%s-%s+%s"%(form[:i], form[i:j], s)
