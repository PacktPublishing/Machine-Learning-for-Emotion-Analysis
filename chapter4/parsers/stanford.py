import re, subprocess

SDPPARSER = "java -mx150m -cp ./*: edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat penn,typedDependencies edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz sdptext"
def sdpparse(s="at least ten commissioners spend time at home.", latex=False):
    s = s.replace(".", " .")
    out = open("sdptext", "w")
    out.write(s)
    out.close()
    x = subprocess.Popen(SDPPARSER.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return stanford2conll(x[0].split(b"\n\n")[1].decode("UTF-8"))
    
dPattern = re.compile("(?P<label>\S*)\((?P<head>\S*), (?P<dtr>\S*)\)")

def stanford2conll(s):
    conll = ""
    for i in dPattern.finditer(s):
        label = i.group("label")
        headWord, headIndex = i.group("head").split("-")
        headIndex = int(headIndex)
        dtrWord, dtrIndex = i.group("dtr").split("-")
        dtrIndex = int(dtrIndex)
        conll += """%s\t%s\t???\t%s\t%s\n"""%(dtrIndex, dtrWord, headIndex, label)
    return conll
