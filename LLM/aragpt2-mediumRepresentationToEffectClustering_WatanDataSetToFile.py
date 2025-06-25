from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
# Load pre-trained model and tokenizer
import import_ipynb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#from google.colab import files
from sklearn.svm import SVC
import gensim
#import tensorflow_text as text
import nltk
import re
import string
#from google.colab import drive
from xlwt import Workbook 

#drive.mount('/content/drive')
import sys
#!pip install pyarabic
import nltk
import re
from nltk.stem.isri import ISRIStemmer
import pandas as pd
#from google.colab import drive
import pyarabic.araby as araby
#!pip install transformers
#!pip install torch
#import torch
from transformers import BertTokenizer, BertForMaskedLM,AutoModelForMaskedLM
from imblearn.over_sampling import SMOTE

from transformers import AutoTokenizer, AutoModel
from imblearn.over_sampling import SMOTE
from collections import Counter


from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tagger.default import DefaultTagger

mled = MLEDisambiguator.pretrained()
tagger = DefaultTagger(mled, 'pos')
def noun(sent):
    y=sent.split()
    nounSen=""
    posT = tagger.tag(sent.split())

    for i in range(len(y)):
        if posT[i]=='noun':
            
            nounSen=nounSen+" "+y[i]
            print("noun")
            print(y[i],' ',posT[i])
    #print(nounSen)        
    return(nounSen)
#drive.mount('/content/drive')
def normalize(sent):

     sent=re.sub('#', '', sent)
     sent=re.sub('_', '', sent)
     sent=re.sub( r'[a-zA-Z0-9]', '', sent)
     sent=re.sub('-', '', sent)
     sent=re.sub('"', '', sent)
     sent=re.sub("'", '', sent)
     sent=re.sub("\.", '', sent)
     sent=re.sub("-", '', sent)
     sent=re.sub("\*", '', sent)
     sent=re.sub("@", '', sent)
     sent=re.sub("\(", '', sent)
     sent=re.sub("\)", '', sent)

     sent=re.sub("[࿐✿❃˺↓]", '', sent)




     sent=araby.strip_tashkeel(sent)
     sent=araby.strip_tatweel(sent)
     word_list=sent.split(' ')
     processed_word_list = []
     for word in word_list:
       word=st.norm(word,3)

       suffix = 'ي'
       if (word.endswith(suffix)):
           word = word[:-1] +'ى'


       suffix = 'ة'
       if (word.endswith(suffix)):
           word = word[:-1] +'ه'
       pref1='ال'

       prefN='ا'
       prefWaw='و'
       if (word.startswith(pref1)  ):
           #print('found al')
           word=word[2:]
       word = re.sub("[إأٱآا]", prefN,word)
       word = re.sub("[وؤ]", prefWaw,word)

      # word=normalizeHashMentionNumber(word)
       processed_word_list.append(word)


     sent=' '.join(processed_word_list)

     return (sent)
def remove_stopwords(sent,stopWordList):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
            word = word.lower() # in case they arenet all lower cased
            if word not in stopWordList:
                processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def remove_shortWords(sent):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
            if len(word) > 2:
                processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def stemOfString(sent):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
                word=st.stem(word)
                if (len(word)>2):
                 processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent
    
def nounString(sent):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
                word=st.stem(word)
                if (len(word)>2):
                 processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def preProcessStem(text):
    text=text.strip()
    text=normalize(text)
    #print("After Normalize")
    #print(text)


    text=remove_stopwords(text,stopWordList)
    #print("After remove_stopwords")

    #print(text)

    text=stemOfString(text)
    #print("After StemOfString")

    #print(text)

    text=remove_shortWords(text)
    #print("After remove_shortWords")

    #print(text)
    text=text.strip()

    return text

def preProcess(text):
    text=text.strip()
    text=normalize(text)
    #print("After Normalize")
    #print(text)


    text=remove_stopwords(text,stopWordList)
    #print("After remove_stopwords")

    #print(text)

    #text=stemOfString(text)
    #print("After StemOfString")

    #print(text)

    text=remove_shortWords(text)
    #print("After remove_shortWords")

    #print(text)
    text=text.strip()

    return text

def get_first_80_words(text):
    # Split the text into words
    words = text.split()
    
    # Get the first 80 words
    first_80_words = words[:80]
    
    # Join the words back into a string
    result = ' '.join(first_80_words)
    
    return result
st = ISRIStemmer()

stopWords=pd.read_excel('RemovedKeywords.xls',0)

stopWordList=stopWords.word

model_name = 'aubmindlab/aragpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
df=pd.read_excel("DataSets/WatanDataSets.xlsx")
content=df["Text"]
url=df['id']
label=df['label']
file= open('aragpt2_WatanDataSets_stem.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        
row=0
#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(1024):
    current_row.append("F"+str(i))
writer.writerow(current_row)
for i in range(len(content)):

   #text = "توليد التضمينات النصية باستخدام ArabianGPT."
   print(i)

   origText=get_first_80_words(content[i])
   text=preProcessStem(origText)
   print(text)
   inputs = tokenizer(text, return_tensors='pt')

   # Forward pass through the model
   outputs = model(**inputs, output_hidden_states=True)

   # The hidden states (embeddings)
   hidden_states = outputs.hidden_states

   # Get the embeddings from the last hidden state
   last_hidden_state = hidden_states[-1]

   # Average of all token embeddings in the last hidden state
   sentence_embedding = torch.mean(last_hidden_state, dim=1)
   print(sentence_embedding)
   print(len(sentence_embedding))
   #matrix.append(sentence_embedding)
   
   
   
   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())
   for x in sentence_embedding:
       print("X")
       print(x)
        # Buffer for the row

       for l in range(len(x)):
         #print(x[l].item())
         current_row.append(x[l].item())
         #sheet1.write(row,j,float(x[l].item()))
         print(current_row)
   writer.writerow(current_row)

   print("Sentence embedding shape:", sentence_embedding.shape)
   print("Sentence embedding:", sentence_embedding)


###############################################################################
file= open('aragpt2_WatanDataSets.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        
row=0
#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(1024):
    current_row.append("F"+str(i))
writer.writerow(current_row)
for i in range(len(content)):

   #text = "توليد التضمينات النصية باستخدام ArabianGPT."
   print(i)

   origText=get_first_80_words(content[i])
   text=preProcess(origText)
   print(text)
   inputs = tokenizer(text, return_tensors='pt')

   # Forward pass through the model
   outputs = model(**inputs, output_hidden_states=True)

   # The hidden states (embeddings)
   hidden_states = outputs.hidden_states

   # Get the embeddings from the last hidden state
   last_hidden_state = hidden_states[-1]

   # Average of all token embeddings in the last hidden state
   sentence_embedding = torch.mean(last_hidden_state, dim=1)
   print(sentence_embedding)
   print(len(sentence_embedding))
   #matrix.append(sentence_embedding)
   
   
   
   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())
   for x in sentence_embedding:
       print("X")
       print(x)
        # Buffer for the row

       for l in range(len(x)):
         #print(x[l].item())
         current_row.append(x[l].item())
         #sheet1.write(row,j,float(x[l].item()))
         print(current_row)
   writer.writerow(current_row)

   print("Sentence embedding shape:", sentence_embedding.shape)
   print("Sentence embedding:", sentence_embedding)

