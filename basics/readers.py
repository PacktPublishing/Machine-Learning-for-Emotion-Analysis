from basics.utilities import *
from basics.corpora import CORPORA, DOWNLOADS
from basics.datasets import BNC, UDT

"""
General purpose data reader. A number of corpora come as collections
of individual files, possibly as the leaves of deeply nested
directories (e.g. the BNC). We deal with these by making a general
purpose reader that crawls through the directory structure looking for
data files (i.e. leaves) and applies a special purpose reader to
these.
"""
                   
def reader(path, dataFileReader, pattern=re.compile(".*"), n=0, N=sys.maxsize, showprogress=False):
    if isinstance(n, int):
        n = [n]
    try:
        pattern.match
    except:
        pattern = re.compile(pattern)
    if n[0] == N:
        return
    if isinstance(path, list):
        # If what you're looking at is a list of file names,
        # look inside it and return the things you find there
        for f in path:
            for r in reader(f, dataFileReader, pattern=pattern, n=n, N=N, showprogress=showprogress):
                if n[0] == N:
                    return
                yield r
    elif os.path.isdir(path):
        # If what you're looking at is a list of file names,
        # look inside it and return the things you find there
        for r in reader([os.path.join(path, f) for f in sorted(os.listdir(path))], dataFileReader, pattern=pattern, n=n, N=N, showprogress=showprogress):
            yield r
    else:
        # If it's a datafile, check that its name matches the pattern
        # and then use the dataFileReader to extract what you want
        if pattern.fullmatch(path):
            for form in dataFileReader(path):
                yield form
                n[0] += 1
                if showprogress and n[0] % 10000 == 0: sys.stdout.write("%s\r"%(n[0]))
                if n[0] == N:
                    return
                
"""
Pattern for splitting the BNC at sentence boundaries
"""
BNCSENTENCE = re.compile("""<s n="(?P<n>\d*)">(?P<text>.*?)</s>""", re.DOTALL)
        
"""
Pattern for getting the form, C5 tag and pos tag from BNC
entries. Easier than using an XML parser.
"""
BNCWordPattern = re.compile("""<(?P<tagtype>w|c) .*?c5="(?P<c5>[A-Z0-9]*?)(-[A-Z0-9]*)?".*?>(?P<form>\S*?)\s*</(?P=tagtype)>""")

def BNCWordReader(data, showprogress=False):
    if showprogress:
        sys.stdout.write("\r%s"%(data))
    for i in BNCWordPattern.finditer(open(data).read()):
        form = i.group("form")
        yield form
        
def BNCTaggedWordReader(data, showprogress=False, N=2):
    if showprogress:
        sys.stdout.write("\r%s"%(data))
    if not "c5" in data:
        data = open(data).read()
    for i in BNCWordPattern.finditer(data):
        form = i.group("form")
        yield (form, i.group("c5")[:N])
        
def BNCTaggedSentenceReader(data, wordreader=BNCTaggedWordReader, showprogress=True, N=sys.maxsize):
    if showprogress:
        sys.stdout.write("\r%s"%(data))
    datafile = open(data)
    data = datafile.read()
    datafile.close()
    for sentence in BNCSENTENCE.finditer(data):
        sentence = list(wordreader(sentence.group("text")))
        yield sentence

def BNCSentenceReader(data, showprogress=False):
    for sentence in BNCTaggedSentenceReader(data, showprogress=showprogress):
        yield " ".join(word[0] for word in sentence)

        
PATBWordPattern = re.compile("INPUT STRING: (?P<form>\S*).*?\* .*?/(?P<tag>[^/\n]*)", re.DOTALL)

def PATBWordReader(path):
    for i in PATBWordPattern.finditer(open(path).read()):
        yield i.group("form")
        
def PATBTaggedWordReader(path):
    for i in PATBWordPattern.finditer(open(path).read()):
        yield i.group("form"), i.group("tag")
 
UDTPattern = re.compile("(?P<COUNTER>\d*)\t(?P<form>\S*)\t(?P<FORM2>\S*)\t(?P<TAG1>\S*)\t(?P<TAG2>\S*)\t(?P<TAG3>\S*)\t(?P<hd>\d*)\t(?P<ROLE>\S*)")

def UTDWordReader(data):
    for i in UDTPattern.finditer(open(data).read()):
        yield i.group("form")

def UDTTaggedWordReader(data):
    for i in UDTPattern.finditer(open(data).read()):
        yield (i.group("form"), i.group("TAG1"))

YourWordPattern = re.compile("...")

def YourWordReader(data):
    for i in UDTPattern.finditer(open(data).read()):
        yield i.group("form")

def YourTaggedWordReader(data):
    for i in UDTPattern.finditer(open(data).read()):
        yield (i.group("form"), i.group("TAG1"))
