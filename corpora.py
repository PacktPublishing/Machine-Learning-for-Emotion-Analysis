from utilities import *
from nltk.corpus import wordnet
import time

CN = "CORPORA/TEXTS/CHINESE"
BNC = "CORPORA/TEXTS/BNC"
NLTKCORPORA = "$HOME/nltk_data/corpora/"
UDT = "CORPORA/TEXTS/ud-treebanks"

"""
The corpora can be downloaded from the following urls (not possible to
write downloaders for some of them, just have to go the site and click
on what you want):

Text corpora
------------
Chinese news stories are from
https://wortschatz.uni-leipzig.de/de/download/Chinese/. This site
provides collections of sentences in a wide range of languages -- the
book just uses 2007-2009 1M and 2020 300K from
https://wortschatz.uni-leipzig.de/de/download/Chinese news, but the
resources for other languages are also potentially useful. These
stories should be stored in CORPORA/TEXTS/CHINESE.

The BNC can be obtained from
https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554. From
there you can get a file called ota_20.500.12024_2554.zip, which
unzips to give you header.xml and 2554.zip, which in turn unzips to a
directory called download which includes Texts which contains the
actual data. There are some broken links if you try to get there from
http://www.natcorp.ox.ac.uk/getting/index.xml -- go to
http://www.natcorp.ox.ac.uk/corpus/index.xml?ID=products to start your
search. download/Texts should be stored as CORPORA/TEXTS/BNC.
 
The Universal Treebank project provides collections of a few hundred
thousand tagged and parsed sentences in a very large number of
languages. These can be downloaded from
https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611 and
should be stored under CORPORA/TEXTS/ud-treebanks. Data for individual
languages can then be accessed by e.g. os.path.join(UDT,
"UD_English").

Tweets corpora
--------------
https://competitions.codalab.org/competitions/17751#learn_the_details-datasets
contains all the data from Semeval-2018. We use the English and Arabic
data from the E-c (emotion classification) task, stored as
CORPORA/TWEETS/SemEval2018-Task1-all-data/Arabic/E-c and
CORPORA/TWEETS/SemEval2018-Task1-all-data/English/E-c

https://github.com/dahouabdelghani/DE-CNN/blob/master/datasets/ASTD.csv
is a collection of Arabic tweets marked as being POS(itive),
NEG(ative) or OBJ(ective). Stored as CORPORA/TWEETS/ASTD.csv
"""

"""
General purpose data reader. A number of corpora come as collections
of individual files, possibly as the leaves of deeply nested
directories (e.g. the BNC). We deal with these by making a general
purpose reader that crawls through the directory structure looking for
data files (i.e. leaves) and applies a special purpose reader to
these.
"""

def leaves(path):
    if os.path.isdir(path):
        l = []
        for f in sorted(os.listdir(path)):
            l += leaves(os.path.join(path, f))
        return l
    else:
        return [path]
    
def reader(path, dataFileReader):
	# If what you're looking at is a directory, look inside it and return the things you find there
    if isinstance(path, list):
        for f in path:
            for r in reader(f, dataFileReader):
                yield r
    else:
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                for r in reader(os.path.join(path, f), dataFileReader):
                    yield r
        else:
            # If it's a datafile, use the dataFileReader to extract what you want
            for r in dataFileReader(path):
                yield r

"""
Patterns used for doing regex-based tokenisation: using these as the
engine for splitting text into tokens is about 3 times as fast the
NLTK word_tokenizer and is easier to adapt to new text genres such as
tweets and to new languages.

Note the use of negative lookahead in ENGLISHPATTERN to catch cases
like "doesn't" and "isn't". The key part of the pattern is
([A-Za-z_](?!n't))*[A-Za-z] -- match as many word characters NOT
FOLLOWED BY "n't" as you can followed by one more word character. That
will pick out "does" from "doesn't" by matching "doe" as characters
not followed by "n't" followed by "s" as the final character.

This is the only really tricky one that is needed for English. Other
languages may require similar use of positive or negative lookahead --
use this one as a model for thinking about such cases.
"""
ENGLISHPATTERN = re.compile(r"""(?P<word>(\d+,?)+(\.\d+)?|(Mr|Mrs|Dr|Prof|St|Rd)\.|n?'\S*|([A-Za-z_](?!n't))*[A-Za-z]|\.|\?|,|\$|£|&|!|"|-|–|[^a-zA-Z\s]+)""")
ARABICPATTERN = re.compile(r"""(?P<word>(\d+,?)+(\.\d+)?|[؟-ۼ]+|\.|\?|,|\$|£|&|!|'|"|\S+)""")
CHINESEPATTERN = re.compile(r"""(?P<word>(\d+,?)+(\.\d+)?|[一-龥]|。|\?|,|\$|£|&|!|'|"|)""")

P = re.compile("([A-Za-z](?!n't))+")

"""
Tokenise the input text using the pattern
"""
def tokenise(text, tokenisePattern=ENGLISHPATTERN):
    return [i.group("word") for i in tokenisePattern.finditer(text.strip())]

"""
Standard NLTK tokeniser for comparison
"""
from nltk.tokenize import word_tokenize

"""
Use the tokeniser as a reader for leaf files, i.e. for tokenising a
corpus
"""
def TokenReader(data, tokenisePattern=CHINESEPATTERN):
	for row in open(data):
		for word in tokenise(row.strip().split("\t")[1], tokenisePattern=tokenizePattern):
			yield word

"""
Pattern for getting the form, C5 tag and pos tag from BNC
entries. Easier than using an XML parser.
"""
BNCWORD = re.compile("""<(?P<tagtype>w|c) .*?c5="(?P<c5>.*?)".*?>(?P<form>.*?)\s*</(?P=tagtype)>""")

"""
Get raw text from BNC leaf files
"""
def BNCWordReader(data):
    print(data)
    for i in BNCWORD.finditer(open(data).read()):
        form = i.group("form")
        if form.lower() == "vh":
            raise Exception(i)
        yield form

"""
Get tagged text from BNC leaf files: you can choose whether to use the
C5 set of tags or the POS set by setting tagype

You can use specials to change the tags of specific words, e.g. use

BNCTaggedWordReader(data, tagtype="c5", specials={"to": "TO"})

to tag every instance of "to" as "TO"

and you can use taglength to convert tags like VVS, NN0 to VV, NN.
"""

def BNCTaggedWordReader(data, tagtype="c5", specials={}, taglength=2):
    if os.path.exists(data):
        print(data)
        data = open(data).read()
    for i in BNCWORD.finditer(data):
        form = i.group("form")
        if form.lower() in specials:
            yield (form, specials[form.lower()])
        else:
            yield (form, i.group(tagtype)[:taglength])

"""
Pattern for splitting the BNC at sentence boundaries
"""
BNCSENTENCE = re.compile("""<s n="(?P<n>\d*)">(?P<text>.*?)</s>""", re.DOTALL)

def BNCTaggedSentenceReader(data, wordreader=BNCTaggedWordReader):
    print(data)
    data = open(data).read()
    for sentence in BNCSENTENCE.finditer(data):
        yield list(wordreader(sentence.group("text")))

def BNCSentenceReader(data):
    for sentence in BNCTaggedSentenceReader(data):
        yield " ".join(word[0] for word in sentence)
        
"""
Get all the distinct words from some resource: using the pattern will
exclude numbers and things like postcodes -- set it to False if you
want absolutely everything, but you probably don't.

l = getLexicon(reader(BNC, BNCWordReader))
"""

AZ = re.compile("^[a-zA-Z]*$")
def getLexicon(wordreader, pattern=AZ):
    l = {}
    for w in wordreader:
        if not pattern or pattern.match(w):
            if w in l:
                l[w] += 1
            else:
                l[w] = 1
    return l

"""
Various useful readers
"""

UDTPattern = re.compile("(?P<COUNTER>\d*)\t(?P<FORM1>\S*)\t(?P<FORM2>\S*)\t(?P<TAG1>\S*)\t(?P<TAG2>\S*)\t(?P<TAG3>\S*)\t(?P<hd>\d*)\t(?P<ROLE>\S*)")
def UDTTaggedWordReader(data):
    for i in UDTPattern.finditer(open(data).read()):
        yield (i.group("FORM1"), i.group("TAG1"))
        
def UDTWordReader(data):
    for i in UDTTaggedWordReader(data):
        yield i[0]
        
def ASTDReader(data):
	for x in open(data):
		x = x.strip().split("\t")
		yield x[0], [x[1]]
    
def SemevalReader(data):
	header = None
	for x in open(data):
		x = x.strip().split("\t")
		if header is None:
			header = x[2:]
		else:
			yield x[1], [h for h, e in zip(header, x[2:]) if e == "1"]
