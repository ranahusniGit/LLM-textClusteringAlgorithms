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
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModelForMaskedLM.from_pretrained("asafaya/bert-base-arabic",
                                  output_hidden_states = True,)

df=pd.read_excel("DataSets/WatanDataSets.xlsx")
content=df["Text"]
url=df['id']
label=df['label']
file= open('bert-base-arabic_WatanDataSets_stem.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        

#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(768):
    current_row.append("F"+str(i))
writer.writerow(current_row)
#for i in range(len(title)):
for i in range(len(content)):

   #text = "توليد التضمينات النصية باستخدام ArabianGPT."
   print(i)

   origText=get_first_80_words(content[i])
   text=preProcessStem(origText)
   print(text)
   print(text)
   marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
   tokenized_text = tokenizer.tokenize(marked_text)

   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

   segments_ids = [1] * len(tokenized_text)

   # Convert inputs to PyTorch tensors
   tokens_tensor = torch.tensor([indexed_tokens])
   segments_tensors = torch.tensor([segments_ids])
        # Run the text through BERT, and collect all of the hidden states produced
         # from all 12 layers.
        #with torch.no_grad():

   outputs = model(tokens_tensor, segments_tensors)
   print("len(outputs)")

   print(len(outputs))
   hidden_states = outputs[1]
   print(hidden_states)
   print(len(hidden_states))
   print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
   layer_i = 0

   print ("Number of batches:", len(hidden_states[layer_i]))
   batch_i = 0

   print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
   token_i = 0

   print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
   # Concatenate the tensors for all layers. We use `stack` here to
   # create a new dimension in the tensor.
   token_embeddings = torch.stack(hidden_states, dim=0)

   print(token_embeddings.size())
        # Remove dimension 1, the "batches".
   token_embeddings = torch.squeeze(token_embeddings, dim=1)

   token_embeddings.size()
        # Swap dimensions 0 and 1.
   token_embeddings = token_embeddings.permute(1,0,2)

   print(token_embeddings.size())
        # Stores the token vectors, with shape [22 x 3,072]
   token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
   for token in token_embeddings:


         cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

         token_vecs_cat.append(cat_vec)
        
# Remove dimension 1, the "batches".
   token_embeddings = torch.squeeze(token_embeddings, dim=1)

   token_embeddings.size()        #print(cls_head.shape ) #hidden states of each [cls]
        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
   token_vecs = hidden_states[-2][0]

        # Calculate the average of all  token vectors.
   sentence_embedding = torch.mean(token_vecs, dim=0)
   
   #matrix.append(sentence_embedding)
   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())

   for x in sentence_embedding:
         # Buffer for the row
         current_row.append(x.item())
         #sheet1.write(row,j,float(x[l].item()))
   writer.writerow(current_row)

   print("Sentence embedding shape:", sentence_embedding.shape)
   print("Sentence embedding:", sentence_embedding)
####################################################################
file= open('bert-base-arabic_WatanDataSets.csv', mode='w', newline='', encoding="utf-8-sig") 
writer = csv.writer(file)        

#for i in range(len(allContent)):
current_row = ['URL','OrigionalText','ProcessedText','Label'] 
for i in range(768):
    current_row.append("F"+str(i))
writer.writerow(current_row)
#for i in range(len(title)):
for i in range(len(content)):

   #text = "توليد التضمينات النصية باستخدام ArabianGPT."
   print(i)

   origText=get_first_80_words(content[i])
   text=preProcess(origText)
   print(text)
   print(text)
   marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
   tokenized_text = tokenizer.tokenize(marked_text)

   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

   segments_ids = [1] * len(tokenized_text)

   # Convert inputs to PyTorch tensors
   tokens_tensor = torch.tensor([indexed_tokens])
   segments_tensors = torch.tensor([segments_ids])
        # Run the text through BERT, and collect all of the hidden states produced
         # from all 12 layers.
        #with torch.no_grad():

   outputs = model(tokens_tensor, segments_tensors)
   print("len(outputs)")

   print(len(outputs))
   hidden_states = outputs[1]
   print(hidden_states)
   print(len(hidden_states))
   print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
   layer_i = 0

   print ("Number of batches:", len(hidden_states[layer_i]))
   batch_i = 0

   print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
   token_i = 0

   print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
   # Concatenate the tensors for all layers. We use `stack` here to
   # create a new dimension in the tensor.
   token_embeddings = torch.stack(hidden_states, dim=0)

   print(token_embeddings.size())
        # Remove dimension 1, the "batches".
   token_embeddings = torch.squeeze(token_embeddings, dim=1)

   token_embeddings.size()
        # Swap dimensions 0 and 1.
   token_embeddings = token_embeddings.permute(1,0,2)

   print(token_embeddings.size())
        # Stores the token vectors, with shape [22 x 3,072]
   token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
   for token in token_embeddings:


         cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

         token_vecs_cat.append(cat_vec)
        
# Remove dimension 1, the "batches".
   token_embeddings = torch.squeeze(token_embeddings, dim=1)

   token_embeddings.size()        #print(cls_head.shape ) #hidden states of each [cls]
        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
   token_vecs = hidden_states[-2][0]

        # Calculate the average of all  token vectors.
   sentence_embedding = torch.mean(token_vecs, dim=0)
   
   #matrix.append(sentence_embedding)
   current_row = [] 
   current_row.append(url[i])
   current_row.append(origText.encode('utf-8').decode())
   current_row.append(text.encode('utf-8').decode())
   current_row.append(label[i].encode('utf-8').decode())

   for x in sentence_embedding:
         # Buffer for the row
         current_row.append(x.item())
         #sheet1.write(row,j,float(x[l].item()))
   writer.writerow(current_row)

   print("Sentence embedding shape:", sentence_embedding.shape)
   print("Sentence embedding:", sentence_embedding)

#fileName="Marocco2016_aragpt2"
#wb.save(fileName+'.xls')   

'''

# Tokenize input text
text = "توليد التضمينات النصية باستخدام ArabianGPT."
inputs = tokenizer(text, return_tensors='pt')

# Forward pass through the model
outputs = model(**inputs, output_hidden_states=True)

# The hidden states (embeddings)
hidden_states = outputs.hidden_states

# Get the embeddings from the last hidden state
last_hidden_state = hidden_states[-1]

# Average of all token embeddings in the last hidden state
sentence_embedding = torch.mean(last_hidden_state, dim=1)

print("Sentence embedding shape:", sentence_embedding.shape)
print("Sentence embedding:", sentence_embedding)
'''
