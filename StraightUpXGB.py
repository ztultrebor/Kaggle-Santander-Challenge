#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

np.random.seed(42)

df_train, df_test = pd.read_csv('train.csv'), pd.read_csv('test.csv')

feature_cols = df_train.columns[1:-1]
X_train = df_train[feature_cols]
y_train = df_train['TARGET']
X_test = df_test[feature_cols]
id_test = df_test['ID']

sd = X_train.std()
empties = sd[sd==0].index
X_train, X_test = X_train.drop(empties,1), X_test.drop(empties,1)

'''
v_cols = None
for col in X_train.columns:
    if not v_cols:
        v_cols = [col]
    else:
        valid = True
        for valid_col in v_cols:
            if all(X_train[col]==X_train[valid_col]):
                valid=False
                break
        if valid:
            v_cols.append(col)
X_train, X_test = X_
train[v_cols], X_test[v_cols]
'''

dependencies = []
feature_cols = X_train.columns
Q, R = np.linalg.qr(np.matrix(X_train))
indep_locs = np.where(abs(R.diagonal())>1e-7)[1]
for i, col in enumerate(feature_cols):
    if i not in indep_locs:
        dependencies.append(col)
X_train, X_test = X_train.drop(dependencies,1), X_test.drop(dependencies,1)
'''
clf = ExtraTreesClassifier()
clf.fit(train, labels)
model = SelectFromModel(clf, prefit=True)
return (pd.DataFrame(model.transform(train)),
        pd.DataFrame(model.transform(test)))
'''

booster = XGBClassifier(
                        n_estimators        =   409,
                        learning_rate       =   0.0202048,
                        max_depth           =   5,
                        subsample           =   0.6815,
                        colsample_bytree    =   0.701
                        )

X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
                                    test_size=0.25, stratify=y_train)

booster.fit(X_fit, y_fit, eval_metric="auc", eval_set=[(X_val, y_val)])

# predicting
y_pred = booster.predict_proba(X_test)[:,1]

submission = pd.DataFrame({'TARGET':y_pred}, index=id_test)
submission.to_csv('submission.csv')

print('Completed!')
