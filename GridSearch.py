#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


def GridSearch(classifier, paramdict, iters, X, y, X_reserve, y_reserve):
    '''
    Takes as input:
        classifier: the sklearn classifier to investigate
        paramdict: a dictionary of parameters to perform the randomized search
        on
        iters: number of iterations of randomized search to perform
        kfcv: a pre-defined StratifiedKFold object
        X: the training data
        y: the target or labels
    What it does:
        acts as a convenent wrapper function for the scikit-learn randomized
        search cross validation method
    Returns:
        the best estimator object
        dictionary of hyperparemeters for the best estimator
        the ROC-AuC score for the best estimator
    '''
    np.random.seed(42)
    kfcv = StratifiedKFold(y, n_folds=5, shuffle=True)

    best_score = 0

    for _ in xrange(iters):
        if 'n_estimators' in paramdict:
            classifier.n_estimators = paramdict['n_estimators'].rvs()
        if 'max_depth' in paramdict:
            classifier.max_depth = paramdict['max_depth'].rvs()
        if 'learning_rate' in paramdict:
            classifier.learning_rate = paramdict['learning_rate'].rvs()
        if 'min_samples_leaf' in paramdict:
            classifier.min_samples_leaf = paramdict['min_samples_leaf'].rvs()
        if 'subsample' in paramdict:
            classifier.subsample = paramdict['subsample'].rvs()
        if 'C' in paramdict:
            classifier.C = paramdict['C'].rvs()
        if 'gamma' in paramdict:
            classifier.gamma = paramdict['gamma'].rvs()
        if 'colsample_bytree' in paramdict:
            classifier.colsample_bytree = paramdict['colsample_bytree'].rvs()
        scores = []
        for fit, val in kfcv:
            classifier.fit(X.iloc[fit], y.iloc[fit])
            scores.append(roc_auc_score(pd.concat([y.iloc[val], y_reserve]), classifier.predict_proba(pd.concat([X.iloc[val], X_reserve]))[:,1]))
        score = 1.*sum(scores)/len(scores)
        if score > best_score:
            print score
            print classifier.get_params()
            best_score = score
            best_params = classifier.get_params()
    best_estimator = classifier.set_params(**best_params)
    return best_estimator, best_params, best_score
