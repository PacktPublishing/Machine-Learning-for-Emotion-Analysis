from basics.utilities import *
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
chars = "A-Za-âzéêèëÇçÀÂÔôàûùîÊï"
chars = "_A-Za-z"
SIMPLESPACES = re.compile(r"""(?P<word>\S+)""")
SIMPLESPACES1 = re.compile(r"""(?P<word>\.|((?!\.)\S)+)""")
ENGLISHPATTERN = re.compile(r"""(?P<word>(\d+,?)+((\.|:)\d+)?K?|(Mr|Mrs|Dr|Prof|St|Rd)\.|n?'t|([%s](?!n't))*[%s]\d*|\.+|\?|,|\$|£|&|:|!|"|/|-|@|#|\(|\)|'(s|re|ll|m|)|[%s]+\d*|\S+)"""%(chars, chars, chars))
ENGLISHPATTERN1 = re.compile(r"""(?P<word>(\d+,?)+((\.|:)\d+)?K?|(Mr|Mrs|Dr|Prof|St|Rd)\.|n?'t|([%s](?!n't))*[%s]\d*|\.+|\?|,|\$|£|&|:|!|"|/|-|@|#|\(|\)|'(s|re|ll|m|)|[%s]+\d*|\S)"""%(chars, chars, chars))
ARABICPATTERN = re.compile(r"""(?P<word>(\d+,?)+(\.\d+)?|[؟-ۼ]+|\.|\?|,|\$|£|&|!|'|"|\S)""")
CHINESEPATTERN = re.compile(r"""(?P<word>(\d+,?)+(\.\d+)?|[一-龥]|。|\?|,|\$|£|&|!|'|"|\S+)""")

P = re.compile("([A-Za-z](?!n't))+")

"""
Tokenise the input text using the pattern
"""
def tokenise(text, tokenisePattern=ENGLISHPATTERN):
    return [i.group("word") for i in tokenisePattern.finditer(text.strip())]

def arabictokenise(text):
    return tokenise(text, tokenisePattern=ARABICPATTERN)

"""
Standard NLTK tokeniser for comparison
"""
from nltk.tokenize import word_tokenize as NLTKtokenise
"""
Use the tokeniser as a reader for leaf files, i.e. for tokenising a
corpus
"""

from basics import dtw
reload(dtw)
def justsplit(s):
    return s.split()

def compare(s, t0, t1):
    t0 = t0(s)
    t1 = t1(s)
    return "<p>%s</p>\n"%(dtw.html(dtw.ARRAY(t0, t1, EXCHANGE=dtw.matchStrings1)))

def compareTweets(tweets, t0, t1, out=sys.stdout):
    if isinstance(out, str):
        out = open(out, "w")
    for tweet in tweets:
        out.write(compare(tweet.src, t0, t1))
    if not out == sys.stdout:
        out.close()
