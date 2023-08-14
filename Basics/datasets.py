#!/usr/local/anaconda3/bin/python

import sys, urllib.request, pathlib, zipfile, shutil, json, tarfile, gzip, glob

from basics.utilities import *
from basics.directories import directories
from basics.corpora import CORPORA, DOWNLOADS

def printemotions(target, emotions):
    return "\t".join("1" if target == emotion else "0" for emotion in emotions)

# unzip src to dest using the appropriate decompressor
def unzip(src, dest):
    if src.endswith(".zip"):
        unpack = zipfile.ZipFile
    elif src.endswith("gz"):
        unpack = tarfile.open
    with unpack(src, "r") as packed:
        packed.extractall(dest)
    
def unzipall(path, dest):
    for d in os.listdir(path):
        d = os.path.join(path, d)
        if os.path.isdir(d):
            unzipall(d, dest)
        else:
            if d.endswith(".zip") or d.endswith("gz"):
                unzip(d, dest)

def cleanup(path, recursive=False, types=re.compile(".*\.((?<!LICENSE\.)txt|(?<![A-Z][A-Z0-9][A-Z0-9]\.)xml|zip|gz|json|jsonl|conll|conllu|md)")):
    for f in os.listdir(path):
        if types.match(f):
            os.remove(os.path.join(path, f))
        elif recursive and os.path.isdir(os.path.join(path, f)):
            cleanup(os.path.join(path, f), recursive=recursive, types=types)
            
class WASSA():
    SRC = "http://saifmohammad.com/WebDocs/EmoInt%%20%s%%20Data"
    LICENCE = "https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html"
    DATA = ["anger-ratings-0to1.%s.txt",
            "fear-ratings-0to1.%s.txt",
            "joy-ratings-0to1.%s.txt",
            "sadness-ratings-0to1.%s.txt",]
    PATH = os.path.join(CORPORA, "TWEETS", "WASSA", "EN")
    DOWNLOAD = os.path.join(DOWNLOADS, "TWEETS", "WASSA", "EN")

    def __init__(self):
        try:
            self.convert()
            cleanup(self.PATH)
        except FileNotFoundError:
            self.checkdownloaded()
        
    def checkdownloaded(self):
        files = []
        urls = []
        for f in self.DATA:
            for t in ["train", "dev.target"]:
                t1 = t.split(".")[0].title()
                f1 = f%(t.lower())
                url = os.path.join(self.SRC%(t1), f1)
                urls.append(url)
                files.append(os.path.join("<CORPORA>", "DOWNLOADS", "TWEETS", "WASSA", "EN", f1))
        print("""You have to download the files 

%s 

to 

%s 

before you can install the WASSA dataset (<CORPORA> is the place where you have chosen to store the datasets: you should set it in corpora.py). Please read the "Terms of Use" at %s before you do so and make sure you adhere to them"""%("\n".join(urls), "\n".join(files), self.LICENCE))

    def convert(self):
        makedirs(self.PATH)
        emotions = set()
        for f in os.listdir(self.DOWNLOAD):
            if f.endswith(".txt"):
                emotions.add(f.split("-")[0])
        emotions = sorted(emotions)
        csv = "ID\ttweet\t%s\n"%("\t".join(emotions))
        for f in os.listdir(self.DOWNLOAD):
            if f.endswith(".txt"):
                emotion = f.split("-")[0]
                for l in open(os.path.join(self.DOWNLOAD, f)):
                    l = l.strip().split("\t")
                    csv += "%s\t%s\t%s\n"%(l[0], l[1], printemotions(emotion, emotions))
        with open(os.path.join(self.PATH, "wholething.csv"), "w") as out:
            out.write(csv)

class SEM4():
            
    SRC = "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/%s/%s"
    PATH = os.path.join(CORPORA, "TWEETS", "SEM4")
    DOWNLOAD = os.path.join(DOWNLOADS, "TWEETS", "SEM4")
    LICENCE = "https://competitions.codalab.org/competitions/17751#learn_the_details-terms_and_conditions"
    zipfile = "%sEI-oc-%s-%s.zip"

    def checkpath(self, l, t):
        return "" if t == "train" and l == "English" else "2018-"
    
    def __init__(self):
        try:
            self.convert()
            cleanup(self.PATH, recursive=True)
        except FileNotFoundError:
            self.checkdownloaded()

    def checkdownloaded(self):
        srcfiles = []
        destfiles = []
        for l in ["English", "Arabic", "Spanish"]:
            l1 = "Es" if l == "Spanish" else l[:2]
            for t in ["dev", "train"]:
                x = self.checkpath(l, t)
                srcfiles.append(self.SRC%(l, self.zipfile%(x, l1, t)))
                destfiles.append(os.path.join("<CORPORA>", "DOWNLOADS", "TWEETS", "SEM11", l1.upper(), self.zipfile%(x, l1, t)))
        print("""
You have to download the files

%s

to

%s

before you can install the SEM4/SEM11 dataset. BE CAREFUL WITH THE NAMES OF THE SOURCE AND DOWNLOADED FILES (particularly the name of the source file for the English training set for the SEM4 set) (<CORPORA> is the place where you have chosen to store the datasets: you should set it in corpora.py).

Please read the terms of use at %s before you do so and make sure you adhere to them
"""%("\n".join(srcfiles), "\n".join(destfiles), self.LICENCE))

    def convert(self, threshold=0):
        emotions = set()
        for l in os.listdir(self.DOWNLOAD):
            unzipall(os.path.join(self.DOWNLOAD, l), os.path.join(self.PATH, l))
        print("Converting data in %s"%(self.PATH))
        for d in os.listdir(self.PATH):
            d = os.path.join(self.PATH, d)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith("txt") and "oc" in f:
                        emotions.add(f.split("-")[-2])
        emotions = sorted(emotions)
        for l in os.listdir(self.PATH):
            d = os.path.join(self.PATH, l)
            if os.path.isdir(d):
                lines0 = []
                for f in os.listdir(d):
                    if f.endswith("txt") and "oc" in f:
                        emotion = f.split("-")[-2]
                        for line in list(open(os.path.join(d, f)))[1:]:
                            line = line.strip().split("\t")
                            try:
                                if int(line[-1][0]) >= threshold:
                                    lines0.append([line[0], line[1], printemotions(line[2], emotions)])
                            except:
                                pass
                csv = "ID\ttweet\t%s\n"%("\t".join(emotions))
                lines1 = []
                for line0 in lines0:
                    for line1 in lines0:
                        if line0[1] in line1[1] and not line0 == line1:
                            break
                    else:
                        lines1.append(line0)
                        csv += "%s\n"%("\t".join(line0))
                makedirs(os.path.join(self.PATH, l))
                with open(os.path.join(self.PATH, l, "wholething.csv"), "w") as out:
                    out.write(csv)

class SEM11(SEM4):
    
    SRC = "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/%s/%s"
    PATH = os.path.join(CORPORA, "TWEETS", "SEM11")
    DOWNLOAD = os.path.join(DOWNLOADS, "TWEETS", "SEM11")
    LDOWNLOAD = os.path.join("<CORPORA>", "DOWNLOADS", "TWEETS", "SEM11")
    zipfile = "%sE-c-%s-%s.zip"

    def checkpath(self, t, l):
        return "2018-"

    def convert(self):
        emotions = set()
        print("Converting data in %s"%(self.DOWNLOAD))
        for d in os.listdir(self.DOWNLOAD):
            d0 = os.path.join(self.DOWNLOAD, d)
            if os.path.isdir(d0):
                csv = ""
                for f in os.listdir(d0):
                    if f.endswith("txt") and "c" in f:
                        lines0 = list(open(os.path.join(d0, f)))
                        if csv == "":
                            csv = lines0[0]
                        for line in lines0[1:]:
                            csv += line
                makedirs(os.path.join(self.PATH, d))
                with open(os.path.join(self.PATH, d, "wholething.csv"), "w") as out:
                    out.write(csv)
                # cleanup(d0)

class CARER():

    SRC = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/data/"
    PATH = os.path.join(CORPORA, "TWEETS", "CARER", "EN")
    DOWNLOAD = os.path.join(DOWNLOADS, "TWEETS", "CARER", "EN")
    LDOWNLOAD = os.path.join("<CORPORA>", "DOWNLOADS", "TWEETS", "CARER", "EN")
    DATA = "data.jsonl.gz"

    def __init__(self):
        try:
            makedirs(self.PATH)
            self.convert()
            cleanup(self.PATH, recursive=True)
        except  FileNotFoundError:
            self.checkdownloaded()

    def checkdownloaded(self):
        print("""
You have to download the file

%s

to

%s

and

%s

to 

%s

before you can install the CARER dataset (<CORPORA> is the place where you have chosen to store the datasets: you should set it in corpora.py).

Please read the terms of use at %s before you do so and make sure you adhere to them
"""%(os.path.join(self.SRC, "dataset_infos.json"), os.path.join(self.LDOWNLOAD, "dataset_infos.json"),os.path.join(self.SRC, self.DATA), os.path.join(self.LDOWNLOAD, self.DATA), "https://huggingface.co/datasets/dair-ai/emotion")) 

    def convert(self):
        with open(os.path.join(self.DOWNLOAD, "dataset_infos.json")) as jsfile:
            infos = json.load(jsfile)
            self.labels = infos["default"]["features"]["label"]["names"]
        with open(os.path.join(self.PATH, "data.jsonl")) as input:
            d = [json.loads(line) for line in input]
        csv = "ID\ttext\t%s\n"%("\t".join(self.labels))
        for i, x in enumerate(d):
            csv += "%s\t%s\t%s\n"%(i, x['text'], "\t".join(["1" if x['label'] == i else "0" for i in range(len(self.labels))]))
        with open(os.path.join(self.PATH, "wholething.csv"), "w") as out:
            out.write(csv)
        
class IMDB():

    SRC = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DOWNLOAD = os.path.join(DOWNLOADS, "TWEETS", "IMDB", "EN")
    LDOWNLOAD = os.path.join("<CORPORA>", "DOWNLOADS", "TWEETS", "IMDB", "EN")
    PATH = os.path.join(CORPORA, "TWEETS", "IMDB", "EN")
    DATA = "aclImdb_v1.tar.gz"

    def __init__(self):
        try:
            self.convert()
            """
            cleanup(self.PATH)
            try:
                shutil.rmtree(os.path.join(self.PATH, "aclImdb"))
            except:
                pass
            """
        except  FileNotFoundError:
            self.checkdownloaded()


    def checkdownloaded(self):
        print("""
Before you can install the IMDB dataset you have to download the file

%s

to

%s"""%(os.path.join(self.SRC), os.path.join(self.LDOWNLOAD, self.DATA)))
              
    def convert(self):
        print("UNZIPPING MAIN FILE")
        unzipall(self.DOWNLOAD, self.PATH)
        csv = "ID\ttweet\tpos\tneg\n"
        print("CONVERTING DATA FILES")
        for d0 in os.listdir(self.PATH):
            d0 = os.path.join(self.PATH, d0)
            if os.path.isdir(d0):
                for d1 in os.listdir(d0):
                    d1 = os.path.join(d0, d1)
                    if os.path.isdir(d1):
                        for d2 in os.listdir(d1):
                            if d2 in ["pos", "neg"]:
                                for f in os.listdir(os.path.join(d1, d2)):
                                    csv += "%s\n"%("\t".join([f, open(os.path.join(d1, d2, f)).read()]+[str(int(d2 == e)) for e in ["pos", "neg"]]))
        with open(os.path.join(self.PATH, "wholething.csv"), "w") as out:
            out.write(csv)

class KWT():
    
    spaces = re.compile("(\s|\\\\)+")
    PATH = os.path.join(CORPORA, "TWEETS", "KWT")

    def __init__(self, subpath):
        self.subpath = subpath
        self.download()
        self.groups = self.norogueannotators()
        self.groups = [g for g in self.groups if len(g) == 3]
        self.convert()
        
    def download(self):
        self.groups = [x.split("\n") for x in open(os.path.join(self.PATH, "original.csv")).read().strip().split("\n\n")]
        self.groups = [[y.split("\t", maxsplit=5) for y in x] for x in self.groups]
        self.convert()

    # Some annotators didn't do very many cases, and
    # some gave 1 for every emotion for some cases.
    # These people also tended not to do the task very well overall,
    # so we eliminate their attributions
    def norogueannotators(self, threshold=8):
        annotators = counter()
        for g in self.groups:
            for x in g:
                annotators.add(x[1])
        return [[x for x in g if annotators[x[1]] > threshold and len(x[2].split("+")) < 5][:3] for g in self.groups]

    def convert(self):
        self.emotions = set()
        for x in self.groups:
            for y in x:
                if y[2] == "":
                    y[2] = set()
                else:
                    y[2] = set(y[2].split("+"))
                    self.emotions = self.emotions.union(y[2])
                y[5] = self.spaces.sub(" ", y[5])
        self.emotions = sorted(self.emotions)
        csv = "id\ttweet\t%s\n"%("\t".join(self.emotions))
        for x in self.groups:
            for y in x:
                y[2] = numpy.array([1 if z in y[2] else 0 for z in self.emotions])
            z = x[0]
            for y in x[1:]:
                z[2] += y[2]
        for x in self.groups:
            csv += "%s\t%s\t%s\n"%(x[0][0], x[0][-1], "\t".join(str(int(e >= self.threshold)) for e in x[0][2]))
        pathlib.Path(self.FULLPATH).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.FULLPATH, "wholething.csv"), "w") as out:
            out.write(csv)
                       
class KWTM(KWT):

    
    def __init__(self):
        self.threshold = 2
        self.FULLPATH = os.path.join(self.PATH, "KWT.M", "AR")
        self.download()
                               
class KWTU(KWT):
    
    def __init__(self):
        self.threshold = 3
        self.FULLPATH = os.path.join(self.PATH, "KWT.U", "AR")
        self.download()

class BNC():

    SRC = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2554/2554.zip?sequence=3&isAllowed=y"
    PATH = os.path.join(CORPORA, "TEXTS", "BNC")
    DOWNLOAD = os.path.join(DOWNLOADS, "TEXTS", "BNC")
    LDOWNLOAD = os.path.join("<CORPORA>", "DOWNLOADS", "TEXTS", "BNC")

    def __init__(self):
        try:
            self.convert()
            cleanup(self.PATH, recursive=True)
        except FileNotFoundError:
            self.checkdownloaded()

    def checkdownloaded(self):print("""You need to download the file 2554.zip from %s to %s before you can install the BNC. Please check the licence details at http://www.natcorp.ox.ac.uk/docs/licence.html before downloading it"""%(self.SRC, self.LDOWNLOAD))

    def convert(self):
        print("""UNZIPPING 2554.zip -- THIS MAY TAKE A FEW MINUTES""")
        unzip(os.path.join(self.DOWNLOAD, "2554.zip"), self.PATH)
                   
class UDT():

    SRC = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5150/ud-treebanks-v2.12.tgz?sequence=1&isAllowed=y"
    PATH = os.path.join(CORPORA, "TREEBANKS", "UDT")
    FULLPATH = os.path.join(PATH, "ud-treebanks-v2.12")
    DOWNLOAD = os.path.join(DOWNLOADS, "TREEBANKS", "UDT")
    LDOWNLOAD = os.path.join("<CORPORA>", "DOWNLOADS", "TREEBANKS", "UDT")
    
    def __init__(self):
        try:
            self.convert()
            cleanup(self.PATH, recursive=True)
            unrestricted, noncommercial = UDT.readlicences()
            print("REMOVING NON-CCAS FILES")
            for f in noncommercial:
                shutil.rmtree(os.path.join(self.FULLPATH, f))
        except FileNotFoundError:
            self.checkdownloaded()

    def checkdownloaded(self):
        print("""
Before you can install the UDT, you have to download the file

%s

to

%s

(<CORPORA> is the place where you have chosen to store the datasets: you should set it in corpora.py).

The UDT consists of a collection of treebanks supplied by different
people, with a variety of licences. Different treebanks from within
the overall collection have different licences. When you download the
main .tgz file you get all of them. This downloader will
remove all except the ones with Creative-Commons-Attribution-ShareAlike
licences (see https://creativecommons.org/licenses/? for more
details) and the .tgz file itself when it finishes.

DO NOT USE THESE TREEBANKS UNLESS YOU AGREE TO THE TERMS OF C-C-A-S LICENCES.
"""%(self.SRC, os.path.join(self.LDOWNLOAD, "ud-treebanks-v2.12.tgz")))
            
    def convert(self):
        print("UNZIPPING THE MAIN SOURCE FILE")
        unzip(os.path.join(self.DOWNLOAD, "ud-treebanks-v2.12.tgz"), self.PATH)
        print("CONVERTING INDIVIDUAL TREEBANKS")
        for which in os.listdir(self.FULLPATH):
            which = os.path.join(self.FULLPATH, which)
            if os.path.isdir(which):
                read_files = glob.glob(os.path.join(which, "*.conllu"))
                with open(os.path.join(which, "wholething"), "wb") as outfile:
                    for f in read_files:
                        with open(f, "rb") as infile:
                            outfile.write(infile.read())

    LICENSEPATTERN = re.compile("(?P<license>Attribution(-NonCommercial)?-ShareAlike|CC BY(-NC)?-SA|General Public License)")
    def readlicences():
        unrestricted = []
        noncommercial = []
        for f in sorted(os.listdir(UDT.FULLPATH)):
            if os.path.isdir(os.path.join(UDT.FULLPATH, f)):
                with open(os.path.join(UDT.FULLPATH, f, "LICENSE.txt")) as license:
                    lines = ""
                    for i, l in enumerate(license):
                        lines += l
                        if i > 3:
                            break
                        m = UDT.LICENSEPATTERN.search(l)
                        if m:
                            if "N" in m.group("license"):
                                noncommercial.append(f)
                            else:
                                unrestricted.append(f)
                            break
        return sorted(unrestricted), sorted(noncommercial)
   
