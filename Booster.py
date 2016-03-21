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

'PCA with leading garbage removal'
from sklearn.decomposition import PCA
n_components =  X_train.shape[1]
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
explained_variance_list = pca.explained_variance_ratio_
leading_garbage = sum([var > 0.0001 for var in explained_variance_list])
print leading_garbage
print sum(pca.explained_variance_ratio_)

X_train = pd.DataFrame(pca.transform(X_train)[:,leading_garbage:], index=X_train.index)
X_test = pd.DataFrame(pca.transform(X_test)[:,leading_garbage:], index=X_test.index)

# Boosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

n_estimators = 500
learning_rate = 10./n_estimators
booster = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

booster.fit(X_train,y_train)

n_folds = 10
scores = np.zeros(n_folds)
cv = KFold(y_train.shape[0], n_folds=n_folds, shuffle=True)
for i, (train, val) in enumerate(cv):
    booster.fit(X_train.iloc[train], y_train.iloc[train])
    scores[i] = roc_auc_score(y_train.iloc[val],
            booster.predict_proba(X_train.iloc[val])[:,1])
print scores
print "AuC-ROC: %0.5f (+/- %0.5f) with %s boosts" % (scores.mean(),
            scores.std() * 2, n_estimators)

# Export results
booster.fit(X_train,y_train)

results = pd.DataFrame({'TARGET':booster.predict_proba(X_test)[:,1]},
            index=test_data['ID'])
print results

results.to_csv('submission.csv')
