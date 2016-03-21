import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib as plt

#read in data
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# break columns into IDs, features and targets
index_col = training_data.columns[0]
feature_cols = training_data.columns[1:-1]
target_col = training_data.columns[-1]
group_size = training_data[training_data['TARGET']==1].shape[0]

#randomize training data for balancing selection
training_data = training_data.reindex(
        np.random.permutation(training_data.index))

#balance the training data
balanced_data = pd.concat(
        [training_data[training_data['TARGET']==1][:group_size],
        training_data[training_data['TARGET']==0][:group_size]])
balanced_data.index = xrange(balanced_data.shape[0])

# here is the data
X_train = balanced_data[feature_cols]
y_train = balanced_data[target_col]
X_test = test_data[feature_cols]

print X_train.shape

# get column metadata
tr_mean_list = X_train.mean()
t_mean_list = X_test.mean()
tr_std_list = X_train.std()
t_std_list = X_test.std()

#remove uniform (typically zero) columns
empties = tr_std_list[tr_std_list==0].index | t_std_list[t_std_list==0].index
X_train = X_train.drop(empties,1)
X_test = X_test.drop(empties,1)

print X_train.shape

#remove duplicate columns
original_and_best = []
tr_mean_std_combo = set()
t_mean_std_combo = set()
feature_cols = X_train.columns[1:-1]
for col in feature_cols:
    if ((tr_mean_list[col], tr_mean_list[col]) not in tr_mean_std_combo and
            (t_mean_list[col], t_mean_list[col]) not in t_mean_std_combo):
        tr_mean_std_combo.add((tr_mean_list[col], tr_mean_list[col]))
        t_mean_std_combo.add((t_mean_list[col], t_mean_list[col]))
        original_and_best.append(col)
X_train = X_train[original_and_best]
X_test = X_test[original_and_best]

print X_train.shape

'''
#remove duplicate columns
dupes = set()
feature_cols = training_data.columns[1:-1]
l = feature_cols.shape[0]
for i in xrange(l):
    i_val_train = training_data[feature_cols[i]].values
    i_val_test = test_data[feature_cols[i]].values
    for j in xrange(i+1, l):
        j_val_train = training_data[feature_cols[j]].values
        j_val_test = test_data[feature_cols[j]].values
        if not ((i_val_train - j_val_train).any()
                or (i_val_test - j_val_test).any()):
            dupes = dupes | set([feature_cols[j]])
training_data = training_data.drop(dupes,1)
test_data = test_data.drop(dupes,1)
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

n_estimators = 64
booster = AdaBoostClassifier(n_estimators=n_estimators)

booster.fit(X_train,y_train)

n_folds = 10
scores = np.zeros(n_folds)
cv = KFold(y_train.shape[0], n_folds=n_folds, shuffle=True)
for i, (train, val) in enumerate(cv):
    booster.fit(X_train.iloc[train], y_train.iloc[train])
    scores[i] = roc_auc_score(y_train.iloc[val],
            booster.predict_proba(X_train.iloc[val])[:,1])
print scores
print "AuC-ROC: %0.5f (+/- %0.5f) with %s boosts" % (scores.mean(), scores.std() * 2, n_estimators)

'''
booster.fit(X_train,y_train)

results = pd.DataFrame({'TARGET':booster.predict_proba(X_test)[:,1]}, index=test_data['ID'])
print results

results.to_csv('submission.csv')
'''
