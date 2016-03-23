# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
#from __future__ import division
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


# load data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
target = 'TARGET'
print df_train.shape

feature_cols = df_train.columns[1:-1]
group_size = df_train[df_train[target]==1].shape[0]

#randomize training data for balancing selection
np.random.seed(49)

df_train = df_train.reindex(
        np.random.permutation(df_train.index))

#balance the training data
df_train_leftover = df_train[df_train[target]==0][group_size:]
df_train = pd.concat(
        [df_train[df_train[target]==1][:group_size],
        df_train[df_train[target]==0][:group_size]])


# get column metadata
tr_mean_list = df_train[feature_cols].mean()
t_mean_list = df_test.mean()
tr_std_list = df_train[feature_cols].std()
t_std_list = df_test.std()

#remove uniform (typically zero) columns
empties = tr_std_list[tr_std_list==0].index | t_std_list[t_std_list==0].index
df_train = df_train.drop(empties,1)
df_test = df_test.drop(empties,1)
df_train_leftover = df_train_leftover.drop(empties,1)
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
X_train, y_train = df_train[original_and_best], df_train[target]
test_ids, X_test = df_test['ID'], df_test[original_and_best]
X_train_leftover, y_train_leftover = df_train_leftover[original_and_best], df_train_leftover[target]
print X_train.shape


# ICA
from sklearn.decomposition import FastICA, PCA
n_components = 55 #X_train.shape[1]
# 32  --> 0.809800
# 48  --> 0.827763
# 52  --> 0.825215
# 54  --> 0.832112
# 55  --> 0.851113
# 56  --> 0.854735
# 64  --> 0.891491
# 128 --> 0.919729
#ica = FastICA(n_components=n_components, whiten=True).fit(X_train)
#X_train = pd.DataFrame(ica.transform(X_train), index=X_train.index)
#X_test = pd.DataFrame(ica.transform(X_test), index=X_test.index)
#X_train_leftover = pd.DataFrame(ica.transform(X_train_leftover),
#        index=X_train_leftover.index)

pca = PCA(n_components=n_components, whiten=True).fit(X_train)
explained_variance_list = pca.explained_variance_ratio_
for col in feature_cols:
    if ((tr_mean_list[col], tr_std_list[col]) not in tr_mean_std_combo and
            (t_mean_list[col], t_std_list[col]) not in t_mean_std_combo):
        tr_mean_std_combo.add((tr_mean_list[col], tr_std_list[col]))
        t_mean_std_combo.add((t_mean_list[col], t_std_list[col]))
        original_and_best.append(col)
X_train, y_train = df_train[original_and_best], df_train[target]
test_ids, X_test = df_test['ID'], df_test[original_and_best]
X_train_leftover, y_train_leftover = df_train_leftover[original_and_best], df_train_leftover[target]
print X_train.shape


# ICA
from sklearn.decomposition import FastICA, PCA
n_components = 55 #X_train.shape[1]
# 32  --> 0.809800
# 48  --> 0.827763
# 52  --> 0.825215
# 54  --> 0.832112
# 55  --> 0.851113
# 56  --> 0.854735
# 64  --> 0.891491
# 128 --> 0.919729
#ica = FastICA(n_components=n_components, whiten=True).fit(X_train)
#X_train = pd.DataFrame(ica.transform(X_train), index=X_train.index)
#X_test = pd.DataFrame(ica.transform(X_test), index=X_test.index)
#X_train_leftover = pd.DataFrame(ica.transform(X_train_leftover),
#        index=X_train_leftover.index)

pca = PCA(n_components=n_components, whiten=True).fit(X_train)
explained_variance_list = pca.explained_variance_ratio_
leading_garbage = sum([var > 0.0001 for var in explained_variance_list])
print leading_garbage
#print sum(pca.explained_variance_ratio_)
X_train = pd.DataFrame(pca.transform(X_train)[:,leading_garbage:],
        index=X_train.index)
X_test = pd.DataFrame(pca.transform(X_test)[:,leading_garbage:],
        index=X_test.index)
X_train_leftover = pd.DataFrame(
        pca.transform(X_train_leftover)[:,leading_garbage:],
        index=X_train_leftover.index)

# booster
n_estimators = 5000
learning_rate = 0.03
max_depth = 3 #5


booster = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.95,
        colsample_bytree=0.85)

X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.1)
# fitting
booster.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric="auc",
        eval_set=[(pd.concat([X_eval,X_train_leftover]), pd.concat([y_eval,
        y_train_leftover]))])

print('Training AUC:', roc_auc_score(pd.concat([y_train, y_train_leftover]),
        booster.predict_proba(pd.concat([X_train,X_train_leftover]))[:,1]))

# predicting
#y_pred= booster.predict_proba(X_test)[:,1]

#submission = pd.DataFrame({target:y_pred}, index=test_ids)
#submission.to_csv('submission.csv')

print('Completed!')
