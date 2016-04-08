#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, r2_score


def GridSearch(classifier, paramdict, iters, X, y, scoring='roc_auc',
                proba=True):
    '''
    Takes as input:
        classifier: the sklearn classifier to investigate
        paramdict: a dictionary of parameters to perform the randomized search
        on
        iters: number of iterations of randomized search to perform
        kfcv: a pre-defined StratifiedKFold object
        X: the training data
        y: the target or labels
        scoring: the type of scoring used to gauge performance. Defaults to
        ROC-AuC
        proba: to calculate probabilities, or not. Defaults to True
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
        if 'intercept_scaling' in paramdict:
            classifier.intercept_scaling = paramdict['intercept_scaling'].rvs()
        if scoring == 'roc_auc':
            scorer = roc_auc_score
        elif scoring == 'f1':
            scorer = f1_score
        elif scoring == 'R2':
            scorer = r2_score
        scores = []
        for fit, val in kfcv:
            classifier.fit(X.iloc[fit], y.iloc[fit])
            if proba:
                scores.append(scorer(y.iloc[val],
                        classifier.predict_proba(X.iloc[val])[:,1]))
            else:
                scores.append(scorer(y.iloc[val],
                        classifier.predict(X.iloc[val])))
        score = 1.*sum(scores)/len(scores)
        if score > best_score:
            print score
            print classifier.get_params()
            best_score = score
            best_params = classifier.get_params()
    best_estimator = classifier.set_params(**best_params)
    return best_estimator, best_params, best_score
