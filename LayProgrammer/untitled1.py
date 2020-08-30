#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:03:23 2020

@author: evkikum
"""


import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import os

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/Practice/NLP/NLP_basics")

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('data/stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features='html5lib')
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')