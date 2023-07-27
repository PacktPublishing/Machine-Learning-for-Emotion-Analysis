import pandas as pd
from nltk.stem.porter import *

emotion_words_file_path = "emotion-words.txt"
data_file_path = "EI-reg-En-anger-train.txt"
stemmer = PorterStemmer()

def stem(sentence):
    res = (" ".join([stemmer.stem(i) for i in sentence.split()]))
    return res

emotion_words  = pd.read_csv(emotion_words_file_path, usecols=[0], names=['word'])
df = pd.read_csv(data_file_path, sep='\t')

# drop rows where the emotion is strong
df[df['Intensity Score'] <= 0.2]

# create some new colums
emotion_words['word_stemmed'] = emotion_words['word']
df['Tweet_stemmed'] = df['Tweet']

# stem the tweets and the emotions words list
df['Tweet_stemmed'] = df['Tweet_stemmed'].apply(stem)
emotion_words['word_stemmed'] = emotion_words['word_stemmed'].apply(stem)

# remove tweets that contain an emotion word
res = []
dropped = []
for _, t_row in df.iterrows():
    tweet = t_row["Tweet_stemmed"]
    add = True
    for _, e_row in emotion_words.iterrows():
        emotion_word = e_row["word_stemmed"]
        if emotion_word in tweet:
            add = False
            break
    if add:
        res.append(t_row["Tweet"])
    else:
        dropped.append(t_row["Tweet"])

for tweet in res:
    print (tweet)

print ("-"*100)

for tweet in dropped:
    print (tweet)
