from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
# Load pre-trained model and tokenizer
import import_ipynb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
import csv
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel

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

stopWordFileName="RemovedKeywords.xls"
stopWords=pd.read_excel('RemovedKeywords.xls',0)

stopWordList=stopWords.word
from transformers import AutoTokenizer, AutoModel
import torch

# Load AraELECTRA model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

df=pd.read_excel("DataSets/WatanDataSets.xlsx")
#allContent=df['Title']+" "+df["Text"]
content=df["Text"]
url=df['id']
label=df['label']

file= open('xlm-roberta-base_WatanDataSets_stem_cls_embedding.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        
file2= open('xlm-roberta-base_WatanDataSets_stem_mean_pooling.csv', mode='w', newline='', encoding="utf-8-sig") 
writer2 = csv.writer(file2)        
#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(768):
    current_row.append("F"+str(i))
writer.writerow(current_row)
writer2.writerow(current_row)

#for i in range(len(title)):
for i in range(len(content)):
   print(i)

   origText=get_first_80_words(content[i])
   text=preProcessStem(origText)  
   inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

   # Get model outputs (this returns all hidden states)
   with torch.no_grad():  # Disable gradient calculations since we're only encoding
       outputs = model(**inputs)

   # Get the last hidden state of the model (tensor of size [batch_size, sequence_length, hidden_size])
   last_hidden_state = outputs.last_hidden_state

   # Method 1: Use the CLS token (the first token) for the sentence representation
   cls_embedding = last_hidden_state[:, 0, :]  # Extract the embedding of the first token ([CLS])

   # Method 2: Mean Pooling - average the token embeddings to get a fixed-size sentence representation
   # Exclude padding tokens in the average pooling if needed
   attention_mask = inputs['attention_mask']
   mean_pooling = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
   
   sentence_embedding = cls_embedding

   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())
   print("len(sentence_embedding)")

   print(len(sentence_embedding))
   for x in sentence_embedding:
        for l in range(len(x)):
          #print(x[l].item())
          current_row.append(x[l].item())
   writer.writerow(current_row)

   sentence_embedding = mean_pooling

   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())
   sentence_embedding = mean_pooling

   for x in sentence_embedding:
         for l in range(len(x)):
           #print(x[l].item())
           current_row.append(x[l].item())
   writer2.writerow(current_row)
###############################################
df=pd.read_excel("DataSets/WatanDataSets.xlsx")
#allContent=df['Title']+" "+df["Text"]
content=df["Text"]
url=df['id']
label=df['label']

file= open('xlm-roberta-base_WatanDataSets_cls_embedding.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        
file2= open('xlm-roberta-base_WatanDataSets_mean_pooling.csv', mode='w', newline='', encoding="utf-8-sig") 
writer2 = csv.writer(file2)        
#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(768):
    current_row.append("F"+str(i))
writer.writerow(current_row)
writer2.writerow(current_row)

#for i in range(len(title)):
for i in range(len(content)):
   #print(i)

   origText=get_first_80_words(content[i])
   text=preProcess(origText)  
   inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

   # Get model outputs (this returns all hidden states)
   with torch.no_grad():  # Disable gradient calculations since we're only encoding
       outputs = model(**inputs)

   # Get the last hidden state of the model (tensor of size [batch_size, sequence_length, hidden_size])
   last_hidden_state = outputs.last_hidden_state

   # Method 1: Use the CLS token (the first token) for the sentence representation
   cls_embedding = last_hidden_state[:, 0, :]  # Extract the embedding of the first token ([CLS])

   # Method 2: Mean Pooling - average the token embeddings to get a fixed-size sentence representation
   # Exclude padding tokens in the average pooling if needed
   attention_mask = inputs['attention_mask']
   mean_pooling = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
   print("mean_pooling")
   print(mean_pooling)

   sentence_embedding = cls_embedding

   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())

   for x in sentence_embedding:
        for l in range(len(x)):
          #print(x[l].item())
          current_row.append(x[l].item())
   writer.writerow(current_row)


   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())
   sentence_embedding = mean_pooling

   for x in sentence_embedding:
         for l in range(len(x)):
           #print(x[l].item())
           current_row.append(x[l].item())
   writer2.writerow(current_row)
