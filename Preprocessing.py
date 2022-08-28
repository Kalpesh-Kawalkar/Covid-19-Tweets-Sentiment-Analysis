import os
import nltk
import numpy as np
import pandas as pd

os.chdir('D:\GCOEN\Data Science with Python\Excel Data Files')
df = pd.read_csv('Corona_NLP.csv', encoding = 'latin1', dtype = str)
df.head()
df.shape
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex

#Removal of Stopwords
sw = stopwords.words('english')
print(sw)
lm = WordNetLemmatizer()

#Preprocessing OriginalTweet
from nltk import word_tokenize, sent_tokenize
tweets = []
t = []
for i in df['OriginalTweet']:
    t = regex.sub('[^A-Za-z0-9]',' ',i)         # Removing Punctuations
    t = t.lower()                               # Conversion To Lowercase
    t = word_tokenize(t)                        # Tokenizing Words
    t = [i for i in t if i not in sw]           # Stopwords Removal
    t = [lm.lemmatize(i) for i in t]            # Lemmatizing The Words
    t = " ".join(t)
    tweets.append(t)

print(tweets[1000])
df['Preprocessed_Tweets'] = tweets
df

df = df.replace({'Sentiment': {'Extremely Positive' : 'Positive', 'Extremely Negative' : 'Negative'}})
pd.set_option("display.max_colwidth", -1)
df
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
sm = cv.fit_transform(df['Preprocessed_Tweets'].iloc[:15000]).toarray()
sm
df['Sentiment'].value_counts()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Sentiment'] = lb.fit_transform(df['Sentiment'])
df['Sentiment'].value_counts()
x = sm
y = df['Sentiment'].iloc[:15000]
print(type(x))
print(type(y))
