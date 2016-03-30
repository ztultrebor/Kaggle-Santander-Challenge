#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dataloader import import_data
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


def best(classifier, paramdict, iters, crsval, X, y):
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
        makes use of the scikit-learn randomized search cross validation
        object. Returns the best estimator fit to the input data, a list of its
        hyperparemeters, and the best ROC-AuC score
    '''
    gs = RandomizedSearchCV(classifier, paramdict, n_iter=iters, cv=crsval,
                            scoring='roc_auc')
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

#===================================prep data==================================

X_train, y_train, X_test, id_test = import_data('train.csv', 'test.csv', 'TARGET', 'ID', verbose=True)

np.random.seed(42)

kfcv = StratifiedKFold(y_train, n_folds=5, shuffle=True)

#==========================Gradient Boost Classifier===========================

params = {'n_estimators': scipy.stats.geom(1/150.),
            'max_depth': scipy.stats.randint(3,10),
            'learning_rate': scipy.stats.expon(0.01),
            'min_samples_leaf': scipy.stats.geom(1/5.),
            'subsample': scipy.stats.beta(2,1)
            }

clf = GradientBoostingClassifier()

best_gb, best_hyperparams, best_gb_auc = best(
                        classifier      =       clf,
                        paramdict       =       params,
                        iters           =       25,
                        crsval          =       kfcv,
                        X               =       X_train,
                        y               =       y_train
            )

print 'Best GB hyperparams: %s' % best_hyperparams
print 'GB AUC: %s' % best_gb_auc

best_gb.fit(X_train, y_train)
# predicting
gb_pred = best_gb.predict_proba(X_test)[:,1]
