from basics.utilities import *
from pathlib import Path

CORPORA = "/Library/WebServer/CGI-Executables/SENTIMENTS/CORPORA"
if not os.path.isdir(CORPORA):
    try:
        os.makedirs(CORPORA)
    except:
        print("""Couldn't make %s as directory to hold corpora -- please set something suitable as the value of CORPORA in basics/corpora.py"""%(CORPORA))
        
DOWNLOADS = os.path.join(CORPORA, "DOWNLOADS")
if not os.path.isdir(CORPORA):
    try:
        os.makedirs(DOWNLOADS)
    except:
        print("""Couldn't make %s as directory to hold downloaded datasets"""%(DOWNLOADS))
    
