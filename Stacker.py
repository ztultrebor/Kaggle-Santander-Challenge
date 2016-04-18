#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dataloader import import_data
import numpy as np
import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score

#==============================================================================

def set_params(classifier, paramdict):
    '''
    Takes as input:
        classifier: an estimator object (scikit-learn compatible)
        paramdict: a dictionary keyed by hyperparameter names with random
        distribution objects as values
    What it does:
        This function is required for grid search over XGB. There is apparently
        some bug in xgbost that causes an error when you call its set_params()
        method
    Returns:
        - an estimator with hyperparemeters fixed
    '''
    for param in paramdict:
        if type(paramdict[param]) in (int, float):
            setattr(classifier, param, paramdict[param])
        else:
            v = paramdict[param].rvs()
            if param in ('max_depth', 'min_samples_leaf', 'n_estimators'):
                setattr(classifier, param, int(v))
            else:
                setattr(classifier, param, v)
    return classifier

#==============================================================================

def shuffle_labels(y_train, folded):
    '''
    Takes as input:
        y_train: a pandas series object contining the training labels/target
        folded: a scikit-learn KFold object
    What it does:
        Reorders the training labels in cross-validated order
    Returns:
        - a pandas series object contining the reordered training labels/target
    '''
    y_train_shuffled = pd.Series()
    for fit, val in folded:
        y_train_shuffled = pd.concat([y_train_shuffled, y_train[val]],
                            ignore_index=True)
    return y_train_shuffled

#==============================================================================

def generalized_CV(method, classifier, paramdict, iters, folds,
                    X_train, y_train, X_test=None):
    '''
    Takes as input:
        classifier: an estimator object (scikit-learn compatible)
        paramdict: a dictionary keyed by hyperparameter names with random
        distribution objects as values
        folds: a scikit-learn KFold cross validation object
        iters: number of estimators to iterate over
        X_train: a pandas DataFrame containing the training data
        y_train: a pandas series containing the target/labels
        method: tells the function how to act: should it perform Grid Search,
        or should it stack or bag?
    What it does:
        Iterates through a sequence of estimators with randomly selected
        hyperparameters. If method=='GridSearch', then it finds the best
        hyperparemeters given the training data. If method=='Stack' or 'Bag'
        then it generates cross validation estimates for the training data and
        fully-trained predictions for the test data using estimators for each
        combination of hyperparameters
    Returns if method=='GridSearch':
        - the best estimator object
        - dictionary of hyperparemeters for the best estimator
        - the ROC-AuC score for the best estimator
    Returns if method=='Stack':
        - a pandas DataFrame containing cross-validation estimates of the
        training labels; each column cotains the estimates for a particular
        estimator
        - a pandas DataFrame containing fully-trained predictions for the test
        data; each column cotains the estimates for a particular estimator
        column cotains the estimates for a particular estimator
        - a pandas series contining the properly ordered training labels/target
        - a list of the hyperparameters for each individual estimator
    Returns if method=='Bag':
        - a pandas DataFrame containing cross-validation estimates of the
        training labels; each column cotains the estimates for a particular
        estimator
        - a pandas DataFrame containing fully-trained predictions for the test
        data; each column cotains the estimates for a particular estimator
        column cotains the estimates for a particular estimator
        - a list of weights for each estimator proportional to that estimator's
        ROC-AuC score
        - a pandas series contining the properly ordered training labels/target
        - a list of the hyperparameters for each individual estimator
    '''
    best_score = 0
    weights = []
    paramlist = []
    y_train_shuffled = shuffle_labels(y_train, folds)
    estimates = pd.DataFrame()
    predictions = pd.DataFrame()
    for _ in xrange(iters):
        esty = set_params(classifier, paramdict)
        training_probs = pd.Series()
        for fit, val in folds:
            # fit this model using this fitting subset
            esty.fit(X_train.iloc[fit], y_train.iloc[fit])
            # predict probs for this validation subset
            val_probs = pd.Series(esty.predict_proba(X_train.iloc[val])[:,1])
            training_probs = pd.concat([training_probs, val_probs],
                                        ignore_index=True)
        score = roc_auc_score(y_train_shuffled, training_probs)
        if method == 'GridSearch':
            if score > best_score:
                best_score = score
                best_params = esty.get_params()
                print score
                print best_params
        elif method in ('Stack', 'Bag'):
            estimates = pd.concat([estimates, training_probs], axis=1,
                                        ignore_index=True)
            # fit this model using full training data
            classifier.fit(X_train, y_train)
            # predict probs for test data
            test_probs = pd.Series(classifier.predict_proba(X_test)[:,1])
            predictions = pd.concat([predictions, test_probs], axis=1,
                                        ignore_index=True)
            params = classifier.get_params()
            paramlist.append(params)
            weights.append((score-0.5)/(0.844-0.5))
            print score
            print params
    if method == 'GridSearch':
        best_estimator = set_params(classifier, best_params)
        # fit training data using best estimator
        best_estimator.fit(X_train, y_train)
        return best_estimator, best_params, best_score
    elif method in ('Stack', 'Bag'):
        return estimates, predictions, weights, y_train_shuffled, params

#===================================prep data==================================

target_col = 'TARGET'
id_col = 'ID'
X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

#================Level 0 Estimator: Gradient Boost Classifier==================

np.random.seed(3)
kfcv0 = StratifiedKFold(y_train, n_folds=4, shuffle=True)
a       =       0
csbt    =       0.7874691933152562
spw     =       5.39673009847897
lr      =       0.040989631409769696
base    =       0.9698413679536542
nest    =       109
sub     =       0.46667628427710284
mcw     =       12.14694715535773
depth   =       5
g       =       0.0960752812134071

params = {
            'n_estimators'      :       scipy.stats.norm(nest, nest/3),
            'learning_rate'     :       scipy.stats.norm(lr, lr/3),
            'max_depth'         :       scipy.stats.norm(depth,depth/3.),
            'subsample'         :       scipy.stats.norm(sub, sub/3),
            'colsample_bytree'  :       scipy.stats.norm(csbt, csbt/3),
            'gamma'             :       scipy.stats.norm(g, g/3),
            'reg_alpha'         :       scipy.stats.norm(a, a/3),
            'scale_pos_weight'  :       scipy.stats.norm(spw, spw/3),
            'min_child_weight'  :       scipy.stats.norm(mcw, mcw/3),
            'base_score'        :       scipy.stats.beta(base/(1-base), 1)
        }

l0Clf = XGBClassifier()
estimates, predictions, weights, y_train, parameters = generalized_CV(
                        method                  =       'Stack',
                        classifier              =       l0Clf,
                        paramdict               =       params,
                        iters                   =       50,
                        folds                   =       kfcv0,
                        X_train                 =       X_train,
                        y_train                 =       y_train,
                        X_test                  =       X_test
            )
estimates.to_csv('./Level1Data/Xtrain.csv', index=False)
pd.DataFrame({target_col:y_train}).to_csv('./Level1Data/ytrain.csv', index=False)
predictions.to_csv('./Level1Data/Xtest.csv', index=False)
pd.DataFrame({id_col:id_test}).to_csv('./Level1Data/idtest.csv', index=False)

#================Level 1 Estimator: Logistic Regression========================

#target_col = 'TARGET'
#id_col = 'ID'
#estimates = pd.read_csv('./Level1Data/Xtrain.csv')
#y_train = pd.read_csv('./Level1Data/ytrain.csv')[target_col]
#predictions = pd.read_csv('./Level1Data/Xtest.csv')
#id_test = pd.read_csv('./Level1Data/idtest.csv')[id_col]
np.random.seed(3)
kfcv1 = StratifiedKFold(y_train, n_folds=5, shuffle=True)
estim_cols = estimates.columns
ROCs = [roc_auc_score(y_train, estimates[col]) for col in estim_cols]

l1Clf = LogisticRegression(max_iter=10000, tol=0.000001,
                            class_weight='balanced')
sorting_hat = zip(estim_cols,ROCs)
sorting_hat.sort(key=lambda x: -x[1])
ordered_cols = [s[0] for s in sorting_hat]
master_cols = []
top_score = 0
for i, result in enumerate(ordered_cols):
    params = {
                'C'                 :       scipy.stats.expon(0, 0.0000005),
                'intercept_scaling' :       scipy.stats.expon(0, 0.01)
                }
    estimator, params, score = generalized_CV(
                    method        =       'GridSearch',
                    classifier    =       l1Clf,
                    paramdict     =       params,
                    iters         =       25,
                    folds         =       kfcv1,
                    X_train       =       estimates[master_cols + [result]],
                    y_train       =       y_train
            )
    if score > top_score:
        top_score = score
        best_params = params
        master_cols.append(result)
    print 'WIP: from %s estimates we choose %s' % (i+1, len(master_cols))
best_estimator = set_params(l1Clf, best_params)
print 'We decided upon %s XGB results' % len(master_cols)

#==============================================================================

print 'The Level1 training data ROC-AuC score is %s' % top_score
best_estimator.fit(estimates[master_cols], y_train)
stacked_prediction = best_estimator.predict_proba(predictions[master_cols])[:,1]
submission = pd.DataFrame({'ID':id_test, 'TARGET':stacked_prediction})
submission.to_csv('submission.csv', index=False)
print 'Completed!'
