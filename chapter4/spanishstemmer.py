import re

SPANISHENDINGS = re.compile("(?<=...)(((?<=a|e|o|i|u)s)|((a|i|e)r)|(a|e|i)(mos|n)|(ái|í|éi)s|((a|o)s?)|é|(a|i)ste|i?ó|(a|i)steis|(a|ie)ron|í)(l(a|o|e)s?)?$")

def stemAll(text):
    if isinstance(text, list):
        return [stemAll(word) for word in text.split(" ")]
    if " " in text:
        return [stemAll(word) for word in text.split(" ")]
    else:
        return SPANISHENDINGS.sub("", text)
