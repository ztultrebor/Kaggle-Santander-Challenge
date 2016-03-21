import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib as plt

#read in data
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#remove uniform (typically zero) columns
tr_std_list = training_data.std()
t_std_list = test_data.std()
empties = tr_std_list[tr_std_list==0].index | t_std_list[t_std_list==0].index
training_data = training_data.drop(empties,1)
test_data = test_data.drop(empties,1)

index_col = training_data.columns[0]
feature_cols = list(training_data.columns[1:-1])
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

X_train = balanced_data[feature_cols]
y_train = balanced_data[target_col]
X_test = test_data[feature_cols]


from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

n_estimators = 64
booster = AdaBoostClassifier(n_estimators=n_estimators)

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
