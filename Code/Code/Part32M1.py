# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from nltk.corpus import stopwords
import base64

path = os.getcwd()
os.chdir(path)
print(path)



'''
with open('test.json', 'r') as myfile:
    data = myfile.read()

# parse file
df = json.loads(data)
#print(df)
'''



df = pd.read_json('train.json')

df.info()

Header1 = 'description'
Header2 = 'test'


def avg_word(data):
  w = data.split()
  return (sum(len(w) for w in w)/len(w))

'''
##Stop word count
SW = stopwords.words('english')

df['StopWord'] = train[Header1].apply(lambda x: len([x for x in x.split() if x in SW]))
df[[Header1,'StopWord']].head()
'''


##Word count
df['word_count'] = df[Header1].apply(lambda x: len(str(x).split(" ") ) )
df[[Header1,'word_count']].head()

##Average word count
df['avg_word'] = df[Header1].apply(lambda x: avg_word(x))
df[[Header1,'avg_word']].head()

##Special Character count
df['Special Characters # @ $'] = df[Header1].apply(lambda x: len([x for x in x.split() if x.startswith('#', '$', '@')]))
df[[Header1,'Special Characters # @ $']].head()

##Most common word
MFreqOcc = pd.Series(' '.join(df[Header1]).split()).value_counts()[:5]
print(MFreqOcc)

##Least common word
LFreqOcc = pd.Series(' '.join(df[Header1]).split()).value_counts()[-20:] #last 20
print(LFreqOcc)
    
pd.Series(' '.join(df.Firm_Name).split()).value_counts()[:3] #top 3

'''



##Number count
df['Numbers'] = df[Header1].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[[Header1,'Numbers']].head()

##Uppercase count
df['Uppercase'] = df[Header1].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[[Header1,'Uppercase']].head()

##Lowercase count
df['Lowercase'] = df[Header1].apply(lambda x: len([x for x in x.split() if x.islower()]))
df[[Header1,'Lowercase']].head()

##Most common word
MFreqOcc = pd.Series(' '.join(df[Header1]).split()).value_counts()[:5]
print(MFreqOcc)

##Least common word
LFreqOcc = pd.Series(' '.join(df[Header1]).split()).value_counts()[-20:]
print(LFreqOcc

##Misspelled word count
#df[Header1][:5].apply(lambda x: str(TextBlob(x).correct()))

      
      
      
      
##Preparing for classification
df['Encoding'] = map(lambda x: x.encode('base64','strict'), df[Header1])

'''

