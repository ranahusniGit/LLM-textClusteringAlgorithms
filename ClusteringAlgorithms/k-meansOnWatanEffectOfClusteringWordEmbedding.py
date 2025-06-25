# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:35:34 2022

@author: user
"""
from transformers import BertTokenizer, BertForMaskedLM,AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel
import gensim
import re
from numpy import array
#from keras.preprocessing.text import one_hot
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import nltk
import re
from nltk.stem.isri import ISRIStemmer
import gensim 
#from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from gensim.models import FastText
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pyarabic.araby as araby
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from keras.layers import Dropout
from keras.layers import GRU
import keras_metrics
import numpy as np
from gensim.test.utils import common_texts
from keras import layers
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from weka.classifiers import Classifier
#from pywekaclassifiers.classifiers import Classifier
#import weka.core.serialization as serialization
import tensorflow as tf
import joblib
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
#import weka.core.jvm as jvm
import pickle
#from weka.core.converters import Loader, Saver
from xlwt import Workbook 
from datetime import date
from datetime import datetime
import torch

from scipy.sparse import csr_matrix
import gensim.downloader as api
from gensim.similarities import WmdSimilarity
from sklearn.metrics.pairwise import euclidean_distances
import xlsxwriter, pandas
from sklearn.feature_extraction.text import  TfidfTransformer





def kmeans_clusters(
	X, 
    k):
   
    km = KMeans(n_clusters=k).fit(X)
    print(f"For n_clusters = {k}")
    

    
    return km, km.labels_



def readBertRepresentation(DataSet):
    features = []
    for i in range(len(DataSet)):
        row=DataSet.iloc[i]
        #print(type(row))
        x=row[4:]
        #print(row)
        #print(x)
        #print(type(x))


        features.append(x) 

    return features;
import os
folder_path="../Watan-2004"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(file_path)
    dataset = pd.read_csv(file_path,sep=',' , encoding='utf-8')
    outFile=filename
    Method="K-means 6 Clusters"
    list_of_docs=dataset.ProcessedText
    urls=dataset.URL
    docs=dataset.OrigionalText
    labels=dataset.Label
    #vectorized_docs = vectorize(urls,docs,list_of_docs, model)
    vectorized_docs=readBertRepresentation(dataset)

    vectorized_docs2=np.asarray( vectorized_docs )

    k=6
    clustering, cluster_labels =kmeans_clusters(
	X=vectorized_docs2,
    k=k
      )
    df_clusters = pd.DataFrame({
     "Url":urls,
     "text":docs ,
     "Processed":list_of_docs,
     "cluster": cluster_labels,
     "Labels":labels,
     })
    df_clusters.to_excel(outFile+"  "+Method+".xlsx", index=False)
