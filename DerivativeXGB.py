#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


# load data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
target_col = 'TARGET'
id_col = 'ID'
print df_train.shape

feature_cols = df_train.columns[1:-1]

# get column metadata
tr_mean_list = df_train[feature_cols].mean()
t_mean_list = df_test.mean()
tr_std_list = df_train[feature_cols].std()
t_std_list = df_test.std()

#remove uniform (typically zero) columns
empties = tr_std_list[tr_std_list==0].index | t_std_list[t_std_list==0].index
df_train = df_train.drop(empties,1)
df_test = df_test.drop(empties,1)
print df_train.shape


#remove duplicate columns
original_and_best = []
tr_mean_std_combo = set()
t_mean_std_combo = set()
feature_cols = df_train.columns[1:-1]
for col in feature_cols:
    if ((tr_mean_list[col], tr_std_list[col]) not in tr_mean_std_combo and
            (t_mean_list[col], t_std_list[col]) not in t_mean_std_combo):
        tr_mean_std_combo.add((tr_mean_list[col], tr_std_list[col]))
        t_mean_std_combo.add((t_mean_list[col], t_std_list[col]))
        original_and_best.append(col)
df_train = df_train[[id_col] + original_and_best + [target_col]]
df_test = df_test[[id_col] + original_and_best]
print df_train.shape


#==============================BALANCING======================================

group_size = 7610 #df_train[df_train[target_col]==1].shape[0]


#randomize training data for balancing selection
np.random.seed(49)

df_train = df_train.reindex(
        np.random.permutation(df_train.index))

#balance the training data
df_train_leftover = df_train[df_train[target_col]==0][group_size:]
df_train = pd.concat(
        [df_train[df_train[target_col]==1][:group_size],
        df_train[df_train[target_col]==0][:group_size]])

X_train, y_train = df_train[original_and_best], df_train[target_col]
test_ids, X_test = df_test['ID'], df_test[original_and_best]
X_train_leftover, y_train_leftover = (df_train_leftover[original_and_best],
        df_train_leftover[target_col])

# booster
n_estimators = 5000
learning_rate = 0.01
max_depth = 5
subsample = 0.6925
CSBT = 0.898


booster = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=CSBT)

X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.3)
# fitting
booster.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric="auc",
        eval_set=[(pd.concat([X_eval,X_train_leftover]), pd.concat([y_eval,
        y_train_leftover]))])

# predicting
y_pred = booster.predict_proba(X_test)[:,1]

submission = pd.DataFrame({target_col:y_pred}, index=test_ids)
submission.to_csv('submission.csv')

print('Completed!')
