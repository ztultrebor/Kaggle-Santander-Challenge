#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./Level1Data/Xtrain.csv')
#X_train['GBpred'] = pd.read_csv('./Level1Data/GBPredtrain.csv')
y_train = pd.read_csv('./Level1Data/ytrain.csv')[target_col]
X_test = pd.read_csv('./Level1Data/Xtest.csv')
#X_test['GBpred'] = pd.read_csv('./Level1Data/GBPredtest.csv')
id_test = pd.read_csv('./Level1Data/idtest.csv')[id_col]


print X_train.shape
print y_train.shape

# booster
n_estimators = 5000
learning_rate = 0.0328
max_depth = 5
subsample = 0.892

booster = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample)

X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
                                    test_size=0.5, stratify=y_train)

booster.fit(X_fit, y_fit, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_val, y_val)])

# predicting
y_pred = booster.predict_proba(X_test)[:,1]

submission = pd.DataFrame({target_col:y_pred}, index=id_test)
submission.to_csv('submission.csv')

print('Completed!')
