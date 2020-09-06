#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:48:28 2020

@author: evkikum
"""


import nltk
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from future.utils import iteritems
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import array
from sklearn.model_selection import train_test_split

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/NLP/NLP_basics/LayProgrammer")

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('data/stopwords.txt'))

positive_reviews = BeautifulSoup(open("electronics/positive.review").read(), features = "html5lib")
positive_reviews = positive_reviews.findAll('review_text')

nagative_reviews = BeautifulSoup(open("electronics/negative.review").read(), features = "html5lib")
nagative_reviews = nagative_reviews.findAll('review_text')


word_index_map = {}
current_index = 0
orig_reviews = []
positive_tokenized = []
nagative_tokenized = []

def my_tokenize(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenize(review.text)
    positive_tokenized.append(tokens)
    
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1
            
        
for review in nagative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenize(review.text)
    nagative_tokenized.append(tokens)
    
    for t in tokens:
        if t not in word_index_map:
            word_index_map[t] = current_index
            current_index += 1
            
   
print("len(word_index_map) ", len(word_index_map))
print("len(positive_tokenized) ", len(positive_tokenized))
print("len(nagative_tokenized) ", len(nagative_tokenized))


df = pd.DataFrame()
df_positive = pd.DataFrame()
df_nagative = pd.DataFrame()

for tokens in positive_tokenized:      
    data = pd.DataFrame()  
    x = np.zeros(len(word_index_map) + 1)    
    for token in tokens:
        i = word_index_map[token]
        x[i] += 1
    x = x/x.sum()
    x[-1] = 1
    data["x"] = x
    data = data.T
    data = data.reset_index()
    data = data.drop(["index"], axis = 1)
    df_positive = df_positive.append(data)
    

df_positive = df_positive.reset_index()
df_positive.info()
df_positive = df_positive.drop(["index"], axis = 1)

for tokens in nagative_tokenized:
    data = pd.DataFrame()
    x = np.zeros(len(word_index_map) + 1)
    
    for token in tokens:
        i = word_index_map[token]
        x[i] += 1
    x = x/x.sum()
    x[-1] = 0
    data["x"] = x
    data = data.T
    data = data.reset_index()
    data = data.drop(["index"], axis = 1)
    df_nagative = df_nagative.append(data)
    
df_nagative = df_nagative.reset_index()
df_nagative.info()
df_nagative = df_nagative.drop(["index"], axis = 1)

df = df.append(df_positive)
df = df.append(df_nagative)
df = df.reset_index()
df = df.drop("index", axis = 1)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

model = LogisticRegression()
model.fit(X, Y)
model.score(X, Y)  ## 78%

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)

model = LogisticRegression()
model.fit(X_train, Y_train)
model.score(X_train, Y_train)  ## 77%
model.score(X_test, Y_test)  ## 70%



    