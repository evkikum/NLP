#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:30:25 2020

@author: evkikum
"""


import pandas as pd
import numpy as np
import nltk # natural language toolkit
#nltk.download()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # TF, TFIDF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

tr_tweets = ['I love this car',
'This view is amazing',
'I feel great this morning',
'I am so excited about the concert',
'He is my best friend',
'I do not like this car',
'This view is horrible',
'I feel tired this morning',
'I am not looking forward to the concert',
'He is my enemy'] #IDVs
tr_sentiment = ["positive","positive","positive","positive","positive",
                "negative","negative","negative","negative","negative"] # DV


## Testing tweets
te_tweets = ['I feel happy this morning', 
'Larry is my friend',
'I do not like that man',
'This view is horrible',
'The house is not great',
'Your song is annoying']
te_sentiment = ["positive","positive","negative","negative","negative","negative"]


output_df = pd.DataFrame(index = ["dt_train","dt_test", "knn_train", "knn_test", "rf_train","rf_test", "gbm_train", "gbm_test", "lr_train", "lr_test"])

twt_vectorizer1 = CountVectorizer(lowercase=False, stop_words=None)
twt_tr_vector1 = twt_vectorizer1.fit(tr_tweets)
twt_tr_vector1_feat = twt_tr_vector1.get_feature_names()
tr_tweets_transformed1 = twt_tr_vector1.transform(tr_tweets)
tr_tweets_transformed1_rawmat = pd.DataFrame(tr_tweets_transformed1.toarray(), columns = twt_tr_vector1.get_feature_names())
te_tweets_transformed1 = twt_tr_vector1.transform(te_tweets)



#### DecisionTreeClassifier
twt_model1 = DecisionTreeClassifier(max_depth = 4, random_state = 1234 ).fit(tr_tweets_transformed1, tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment), sent_pre_tr_model1)
dt_train = accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 100%
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment), sent_pre_te_model1)
dt_test = accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 83%


twt_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(tr_tweets_transformed1, tr_sentiment)
twt_pred_tr = twt_knn1.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
knn_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_knn1.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
knn_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66%


twt_rf = RandomForestClassifier().fit(tr_tweets_transformed1, tr_sentiment)
twt_pred_tr = twt_rf.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
rf_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_rf.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
rf_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 100%


twt_gbm = GradientBoostingClassifier().fit(tr_tweets_transformed1, tr_sentiment)
twt_pred_tr = twt_gbm.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
gbm_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_gbm.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
gbm_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 83%


twt_lr = LogisticRegression().fit(tr_tweets_transformed1, tr_sentiment)
twt_pred_tr = twt_lr.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
lr_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_lr.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
lr_test = accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 83%

output_df["CV"] = [dt_train,dt_test, knn_train, knn_test, rf_train,rf_test, gbm_train, gbm_test, lr_train, lr_test]


##################################################################################
twt_vectorizer2 = CountVectorizer(lowercase=True, stop_words='english')
twt_tr_vector2 = twt_vectorizer2.fit(tr_tweets)
twt_tr_vector2_feat = twt_tr_vector2.get_feature_names()
tr_tweets_transformed2 = twt_tr_vector2.transform(tr_tweets)
tr_tweets_transformed2_rawmat = pd.DataFrame(twt_tr_vector2.transform(tr_tweets).toarray(), columns = twt_tr_vector2.get_feature_names())
te_tweets_transformed2 = twt_tr_vector2.transform(te_tweets)

#### DecisionTreeClassifier
twt_model1 = DecisionTreeClassifier().fit(tr_tweets_transformed2, tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment), sent_pre_tr_model1)
dt_train = accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 90%
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment), sent_pre_te_model1)
dt_test = accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 50 %


twt_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(tr_tweets_transformed2, tr_sentiment)
twt_pred_tr = twt_knn1.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
knn_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)
twt_pred_te = twt_knn1.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 83 %
knn_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 83 %

twt_rf = RandomForestClassifier().fit(tr_tweets_transformed2, tr_sentiment)
twt_pred_tr = twt_rf.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
rf_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_rf.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 100%
rf_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 100%


twt_gbm = GradientBoostingClassifier().fit(tr_tweets_transformed2, tr_sentiment)
twt_pred_tr = twt_gbm.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
gbm_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_gbm.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66%
gbm_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66%



twt_lr = LogisticRegression().fit(tr_tweets_transformed2, tr_sentiment)
twt_pred_tr = twt_lr.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
lr_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_lr.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%
lr_test = accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%


output_df["CV_LC_SW"] = [dt_train,dt_test, knn_train, knn_test, rf_train,rf_test, gbm_train, gbm_test, lr_train, lr_test]


##################################################################################


twt_vectorizer3 = CountVectorizer(lowercase = True, stop_words = "english",
                                  ngram_range = (1,2),max_features = 100)
twt_tr_vector3 = twt_vectorizer3.fit(tr_tweets)
twt_tr_vector3_feat = twt_tr_vector3.get_feature_names()
tr_tweets_transformed3 = twt_tr_vector3.transform(tr_tweets)
te_tweets_transformed3 = twt_tr_vector3.transform(te_tweets) 


#### DecisionTreeClassifier

twt_model1 = DecisionTreeClassifier().fit(tr_tweets_transformed3, tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment), sent_pre_tr_model1)
accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 100%
dt_train = accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 100%
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment), sent_pre_te_model1)
accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 66 %
dt_test = accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 66 %


twt_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(tr_tweets_transformed3, tr_sentiment)
twt_pred_tr = twt_knn1.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
knn_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_knn1.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 100 %
knn_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 100 %

twt_rf = RandomForestClassifier().fit(tr_tweets_transformed3, tr_sentiment)
twt_pred_tr = twt_rf.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
rf_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_rf.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66%
rf_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66%


twt_gbm = GradientBoostingClassifier().fit(tr_tweets_transformed3, tr_sentiment)
twt_pred_tr = twt_gbm.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
gbm_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_gbm.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 50%
gbm_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 50%

twt_lr = LogisticRegression().fit(tr_tweets_transformed3, tr_sentiment)
twt_pred_tr = twt_lr.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
lr_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_lr.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%
lr_test = accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%

output_df["CV_LC_SW_NG"] = [dt_train,dt_test, knn_train, knn_test, rf_train,rf_test, gbm_train, gbm_test, lr_train, lr_test]

##################################################################

twt_vectorizer4 = TfidfVectorizer(lowercase=True, stop_words = 'english', max_features = 100)
twt_tr_vector4 = twt_vectorizer4.fit(tr_tweets)
twt_tr_vector4_feat = twt_tr_vector4.get_feature_names()
tr_tweets_transformed4 = twt_tr_vector4.transform(tr_tweets)

te_tweets_transformed4 = twt_tr_vector4.transform(te_tweets)


#### DecisionTreeClassifier
twt_model1 = DecisionTreeClassifier().fit(tr_tweets_transformed4, tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment), sent_pre_tr_model1)
accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 100%
dt_train = accuracy_score(np.array(tr_sentiment), sent_pre_tr_model1)  ## 100%
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment), sent_pre_te_model1)
accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 66 %
dt_test = accuracy_score(np.array(te_sentiment), sent_pre_te_model1)   ## 66 %


twt_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(tr_tweets_transformed4, tr_sentiment)
twt_pred_tr = twt_knn1.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
knn_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_knn1.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66 %
knn_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66 %


twt_rf = RandomForestClassifier().fit(tr_tweets_transformed4, tr_sentiment)
twt_pred_tr = twt_rf.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
rf_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_rf.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 83%
rf_test = accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 83%


twt_gbm = GradientBoostingClassifier().fit(tr_tweets_transformed4, tr_sentiment)
twt_pred_tr = twt_gbm.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
gbm_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_gbm.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66.66 %
gbm_test= accuracy_score(np.array(te_sentiment), twt_pred_te)  ## 66.66 %

twt_lr = LogisticRegression().fit(tr_tweets_transformed4, tr_sentiment)
twt_pred_tr = twt_lr.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment), twt_pred_tr)
accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
lr_train = accuracy_score(np.array(tr_sentiment), twt_pred_tr)  ## 100%
twt_pred_te = twt_lr.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment), twt_pred_te)
accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%
lr_test = accuracy_score(np.array(te_sentiment), twt_pred_te)   ## 66%

output_df["TFIDF"] = [dt_train,dt_test, knn_train, knn_test, rf_train,rf_test, gbm_train, gbm_test, lr_train, lr_test]