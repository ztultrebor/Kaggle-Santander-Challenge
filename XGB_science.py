#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dataloader import import_data
import numpy as np
import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split
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
    if 'n_estimators' in paramdict:
        classifier.n_estimators = paramdict['n_estimators']
    if 'max_depth' in paramdict:
        classifier.max_depth = paramdict['max_depth']
    if 'learning_rate' in paramdict:
        classifier.learning_rate = paramdict['learning_rate']
    if 'min_samples_leaf' in paramdict:
        classifier.min_samples_leaf = paramdict['min_samples_leaf']
    if 'subsample' in paramdict:
        classifier.subsample = paramdict['subsample']
    if 'C' in paramdict:
        classifier.C = paramdict['C']
    if 'gamma' in paramdict:
        classifier.gamma = paramdict['gamma']
    if 'colsample_bytree' in paramdict:
        classifier.colsample_bytree = paramdict['colsample_bytree']
    if 'intercept_scaling' in paramdict:
        classifier.intercept_scaling = paramdict['intercept_scaling']
    if 'base_score' in paramdict:
        classifier.base_score = paramdict['base_score']
    if 'scale_pos_weight' in paramdict:
        classifier.scale_pos_weight = paramdict['scale_pos_weight']
    if 'min_child_weight' in paramdict:
        classifier.min_child_weight = paramdict['min_child_weight']
    return classifier

#==============================================================================

def randomize_params(classifier, paramdict):
    '''
    Takes as input:
        classifier: an estimator object (scikit-learn compatible)
        paramdict: a dictionary keyed by hyperparameter names with random
        distribution objects as values
    What it does:
        Associates each hyperparameter in the paramdict andomly selects hyperparameters for the chosen estimator.
    Returns:
        - an estimator with hyperparemeters fixed
    '''
    if 'n_estimators' in paramdict:
        classifier.n_estimators = int(paramdict['n_estimators'].rvs())
    if 'max_depth' in paramdict:
        classifier.max_depth = int(paramdict['max_depth'].rvs())
    if 'learning_rate' in paramdict:
        classifier.learning_rate = paramdict['learning_rate'].rvs()
    if 'min_samples_leaf' in paramdict:
        classifier.min_samples_leaf = int(paramdict['min_samples_leaf'].rvs())
    if 'subsample' in paramdict:
        classifier.subsample = paramdict['subsample'].rvs()
    if 'C' in paramdict:
        classifier.C = paramdict['C'].rvs()
    if 'gamma' in paramdict:
        classifier.gamma = paramdict['gamma'].rvs()
    if 'colsample_bytree' in paramdict:
        classifier.colsample_bytree = paramdict['colsample_bytree'].rvs()
    if 'intercept_scaling' in paramdict:
        classifier.intercept_scaling = paramdict['intercept_scaling'].rvs()
    if 'base_score' in paramdict:
        classifier.base_score = paramdict['base_score'].rvs()
    if 'scale_pos_weight' in paramdict:
        classifier.scale_pos_weight = paramdict['scale_pos_weight'].rvs()
    if 'min_child_weight' in paramdict:
        classifier.min_child_weight = paramdict['min_child_weight'].rvs()
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
                    X_train, y_train, X_test=None, best_score=0):
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
    originals = classifier.get_params()
    weights = []
    paramlist = []
    y_train_shuffled = shuffle_labels(y_train, folds)
    estimates = pd.DataFrame()
    predictions = pd.DataFrame()
    improved = False
    for _ in xrange(iters):
        esty = randomize_params(classifier, paramdict)
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
                improved = True
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
            if method == 'Bag':
                weights.append((score-0.5)/(0.844-0.5))
            print score
            print params
    if method == 'GridSearch':
        if improved:
            best_estimator = set_params(classifier, best_params)
            # fit training data using best estimator
            best_estimator.fit(X_train, y_train)
            return best_estimator, best_params, best_score
        else:
            return set_params(classifier, originals), originals, best_score
    elif method == 'Stack':
        return estimates, predictions, y_train_shuffled, params
    elif method == 'Bag':
        return estimates, predictions, weights, y_train_shuffled, params

#===================================prep data==================================

target_col = 'TARGET'
id_col = 'ID'
X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

np.random.seed(3)
kfcv0 = StratifiedKFold(y_train, n_folds=4, shuffle=True)
#X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
#                                    test_size=0.25, stratify=y_train)


#X_train = pd.read_csv('./Level1Data/Xtrain.csv')
#y_train = pd.read_csv('./Level1Data/ytrain.csv')[target_col]

#=======================Initial Guess at Hyperparameters=======================

params =            {       'max_depth'           :       5,
                            'min_child_weight'    :       1,
                            'gamma'               :       1,
                            'n_estimators'        :       50,
                            'learning_rate'       :       0.1,
                            'subsample'           :       0.8,
                            'reg_alpha'           :       1,
                            'colsample_bytree'    :       0.8,
                            'base_score'          :       0.5,
                            'scale_pos_weight'    :       10
                    }
Clf = set_params(XGBClassifier(), params)

score = 0

while True:
    #=============Tuning Round 1: max_depth & min_child_weight=================

    print 'Grid Searching max_depth and min_child_weight'
    depth = params['max_depth']
    mcw = params['min_child_weight']
    params = {
                'max_depth'         :       scipy.stats.norm(depth, depth/3),
                'min_child_weight'  :       scipy.stats.expon(0, mcw)
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       25,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with max_depth and min_child_weight'

    #=========================Tuning Round 2: gamma============================

    print 'Grid Searching gamma'
    g = params['gamma']
    params = {
                'gamma'             :       scipy.stats.expon(0, g)
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       5,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with gamma'

    #==============Tuning Round 3: subsample and colsample_bytree==============

    print 'Grid Searching subsample and colsample_bytree'
    sub = params['subsample']
    csbt = params['colsample_bytree']
    params = {
                'subsample'         :       scipy.stats.beta(sub/(1-sub),1),
                'colsample_bytree'  :       scipy.stats.beta(csbt/(1-csbt),1)
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       25,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with subsample and colsample_bytree'

    #==========================Tuning Round 4: alpha===========================

    print 'Grid Searching alpha'
    a = params['reg_alpha']
    params = {
                'reg_alpha'         :       scipy.stats.expon(0, a)
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       5,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with alpha'

    #============Tuning Round 5: n_estimators and learning_rate================

    print 'Grid Searching n_estimators and learning_rate'
    nest = params['n_estimators']
    lr = params['learning_rate']
    params = {
                'n_estimators'      :       scipy.stats.expon(0, nest),
                'learning_rate'     :       scipy.stats.expon(0, lr),
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       25,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with n_estimators and learning_rate'

    #============Tuning Round 6: base score and scale_pos_weight===============

    print 'Grid Searching base_score and scale_pos_weight'
    base = params['base_score']
    params = {
                'base_score'        :       scipy.stats.beta(1,1),
                'scale_pos_weight'  :       scipy.stats.expon(0, 10),
                }
    estimator, params, score = generalized_CV(
                            method                  =       'GridSearch',
                            classifier              =       Clf,
                            paramdict               =       params,
                            iters                   =       25,
                            folds                   =       kfcv0,
                            X_train                 =       X_train,
                            y_train                 =       y_train,
                            X_test                  =       X_test,
                            best_score              =       score
                )
    Clf = estimator
    print 'Done with base_score and scale_pos_weight'
