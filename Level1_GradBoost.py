#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

#===================================prep data==================================

np.random.seed(42)

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

#=============================================================================

np.random.seed(42)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
            test_size=0.66667, stratify=y_train)

#==========================Gradient Boost Classifier===========================

gb = GradientBoostingClassifier(
                n_estimators      =       160,
                subsample         =       0.892,
                learning_rate     =       0.0328,
                max_depth         =       5,
                min_samples_leaf  =       20
        )

gb_auc = 0.838492734703 #from GradientBooster.py
gb.fit(X_fit, y_fit)
print 'Fit to fitting data is completed'
gb_val_pred = gb.predict_proba(X_val)[:,1]
gb_auc = roc_auc_score(y_val, gb_val_pred)

print 'Gradient Booster ROC-AuC is %s' % gb_auc

gb.fit(X_train, y_train)
print 'Fit to training data is completed'
# predicting
gb_pred = gb.predict_proba(X_test)[:,1]
print 'Prediction on test data is completed'

X_val.to_csv('./Level1Data/Xtrain.csv', index=False)
pd.Series(y_val).to_csv('./Level1Data/ytrain.csv', index=False)
X_test.to_csv('./Level1Data/Xtest.csv', index=False)
pd.Series(id_test).to_csv('./Level1Data/idtest.csv', index=False)
pd.Series(gb_val_pred).to_csv('./Level1Data/GBPredtrain.csv', index=False)
pd.Series(gb_pred).to_csv('./Level1Data/GBPredtest.csv', index=False)

print('We done here yo')
