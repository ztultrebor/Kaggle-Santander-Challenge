#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

#===================================prep data==================================

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

#=============================================================================

np.random.seed(42)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
            test_size=0.75, stratify=y_train)

#==========================Gradient Boost Classifier===========================

clf = LogisticRegression(
                C                       =       0.757031493720797,
                intercept_scaling       =       0.18989143560912067
        )
#0.609381720696

clf.fit(X_fit, y_fit)
print 'Fit to fitting data is completed'
clf_val_pred = clf.predict_proba(X_val)[:,1]
clf_auc = roc_auc_score(y_val, clf_val_pred)

print 'Logistic Regression ROC-AuC is %s' % clf_auc

clf.fit(X_train, y_train)
print 'Fit to training data is completed'
# predicting
clf_pred = clf.predict_proba(X_test)[:,1]
print 'Prediction on test data is completed'

X_val.to_csv('./Level1Data/Xtrain.csv', index=False)
pd.DataFrame({target_col:y_val}).to_csv('./Level1Data/ytrain.csv', index=False)
X_test.to_csv('./Level1Data/Xtest.csv', index=False)
pd.DataFrame({id_col:id_test}).to_csv('./Level1Data/idtest.csv', index=False)
pd.DataFrame({'LogisticPob':clf_val_pred}).to_csv(
                './Level1Data/LogisticTrain.csv', index=False)
pd.DataFrame({'LogisticProb':clf_pred}).to_csv(
                './Level1Data/LogisticTest.csv', index=False)

print('We done here yo')
