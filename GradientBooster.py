#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dataloader import import_data
from GridSearch import GridSearch
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

#===================================prep data==================================

X_train, y_train, X_test, id_test = import_data('train.csv', 'test.csv', 'TARGET', 'ID', verbose=True)

#==========================Gradient Boost Classifier===========================

params = {'n_estimators': scipy.stats.geom(1/150.),
            'max_depth': scipy.stats.randint(3,10),
            'learning_rate': scipy.stats.expon(0.01),
            'min_samples_leaf': scipy.stats.geom(1/5.),
            'subsample': scipy.stats.beta(2,1)
            }

clf = GradientBoostingClassifier()

best_gb, best_hyperparams, best_gb_auc = GridSearch(
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
