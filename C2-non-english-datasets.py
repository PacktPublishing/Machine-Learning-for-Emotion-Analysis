import translators as ts

phrase = 'The quick brown fox jumps over the lazy dog.'
phrase = 'توتر فز قلبي'

FROM_LANG = 'ar'
TO_LANG = 'en'

ts.google
res = ts.google(phrase, from_language=FROM_LANG, to_language=TO_LANG)
print (res)

ts.baidu
res = ts.baidu(phrase, from_language='ara', to_language=TO_LANG)
print (res)

ts.alibaba
res = ts.alibaba(phrase, from_language=FROM_LANG, to_language=TO_LANG)
print (res)

ts.bing
res = ts.bing(phrase, from_language=FROM_LANG, to_language=TO_LANG)
print (res)
