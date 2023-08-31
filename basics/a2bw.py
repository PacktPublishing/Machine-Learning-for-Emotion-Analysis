#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import sys

a2bwtable = {"ا" : "A",
             "أ" : "O",
             "ب" : "b",
             "ت" : "t",
             "ث" : "v",
             "ج" : "j",
             "ح" : "H",
             "خ" : "x",
             "د" : "d",
             "ذ" : "*",
             "ر" : "r",
             "ز" : "z",
             "س" : "s",
             "ش" : "$",
             "ص" : "S",
             "ض" : "D",
             "ط" : "T",
             "ظ" : "Z",
             "ع" : "E",
             "غ" : "g",
             "ف" : "f",
             "ق" : "q",
             "ك" : "k",
             "ل" : "l",
             "م" : "m",
             "ن" : "n",
             "و" : "w",
             "ى" : "Y",
             "ه" : "h",
             "ي" : "y",
             "آ" : "|",
             "ؤ" : "W",
             "ئ" : "}",
             "ة" : "p",
             "َ" : "a",
             "ُ" : "u",
             "ِ" : "i",
             "ّ" : "~",
             "ْ" : "o",
             "إ" : "I",
             "ء" : "Q",
             "ـ" : "" ,
             "ً" : "F",
             "ٌ" : "N",
             "ٍ" : "K",
             "٠" : "0",
             "١" : "1",
             "٢" : "2",
             "٣" : "3",
             "٤" : "4",
             "٥" : "5",
             "٦" : "6",
             "٧" : "7",
             "٨" : "8",
             "٩" : "9",
             ";" : ";",
             "£" : "P",
             "؟" : "?" ,
             "،" : "," ,
             "…" : "." ,
             "؛" : ";" ,
             "-" : "-",
             "--" : "--",
             "/" : "/",
             "“" : '"',
             "”" : '"',
             "!" : "!",
             ":" : ":",
             ".." : "..",
             "..." : "...",
             "^" : "^",
             " " : " "
         }

"""
s = 'بتوقيت'
teststring = 'بتوقيت'
print teststring
"""

bw2atex = {'*': '_d', 'E': '`', '$': '^s', 'g': '.g', 'I': "'i", 'H': '.h', 'j': '^g', 'O': "'a", 'p': 'T', 'S': '.s', 'T': '.t', 'W': "'w", 'v': '_t', 'x': '_h', '{': '"', 'Z': '.z', '}': "'y", '|': "'A", 'D': '.d'}

def invtable(t0):
     t1 = {}
     for x in t0:
          t1[t0[x]] = x
     return t1

bw2atable = invtable(a2bwtable)

def convert(s0, table=a2bwtable):
    s1 = u''
    for c in s0:
        try:
            s1 += table[c]
        except:
            try:
                s1 += c
            except:
                s1 += '?'
    try:
        return str(s1)
    except:
        try:
            return str(codecs.encode(s1, "utf-8"))
        except:
            return "???"
    
def r2l(s0):
    s1 = u""
    for c in s0:
        s1 = c+s1
    return s1

def doConversion(dir, src, out=sys.stdout):
    try:
        if dir == "a2bw":
            with safeout(out) as write:
                write(convert(codecs.open(src, encoding="utf-8").read(), a2bwtable))
        elif dir == "bw2a":
            with safeout(out, encoding="utf-8") as write:
                write(convert(open(src).read(), bw2atable).decode("utf-8"))
        else:
            raise Exception("direction has to be either a2bw or bw2a")
    except Exception as e:
        print(e)

if "a2bw.py" in sys.argv[0]:
    for kv in sys.argv[1:]:
        k, v = kv.split("=")
        if "direction".startswith(k):
            dir = v
        elif "src".startswith(k):
            src = v
        elif "dest".startswith(k):
            dest = v
    try:
        doConversion(dir, src, dest)
    except Exception as e:
        print(e)
        
