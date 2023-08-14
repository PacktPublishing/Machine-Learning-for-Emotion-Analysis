from utilities import *
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
chars = "A-Za-z"
ENGLISHPATTERN = re.compile(r"""(?P<word>(\d+,?)+((\.|:)\d+)?K?|(Mr|Mrs|Dr|Prof|St|Rd)\.|n?'t|([%s](?!n't))*[%s]|\.|\?|,|\$|£|&|:|!|"|-|\S)"""%(chars, chars))
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
from nltk.tokenize import word_tokenize
"""
Use the tokeniser as a reader for leaf files, i.e. for tokenising a
corpus
"""
