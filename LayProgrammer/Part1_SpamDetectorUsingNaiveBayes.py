#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:33:41 2020

@author: evkikum
"""




from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import os

# Note: technically multinomial NB is for "counts", but the documentation says
#       it will work for other types of "counts", like tf-idf, so it should
#       also work for our "word proportions"

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/Practice/NLP/NLP_basics")

data = pd.read_csv("data/spambase.data").values
np.random.shuffle(data)

X = data[:,:48]
Y = data[:,-1]

Xtrain =  X[:-100,]
Ytrain = Y[:-100,]

##Last 100 will be the test
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print("Classification rate for NB:", model.score(Xtest, Ytest))  ## 89%

##### you can use ANY model! #####
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)

print("Classification rate for AdaBost:", model.score(Xtest, Ytest))  ## 96%  

## IN BOTH THE ABOVE MODELS IF SCORE > 80 THEN IT IS GOOD MODEL AND CONSIDERED AS SPAM.

