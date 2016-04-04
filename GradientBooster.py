#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dataloader import import_data
from GridSearch import GridSearch
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingClassifier


#===================================prep data==================================

np.random.seed(42)

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

#==========================Gradient Boost Classifier===========================

params = {
            'n_estimators'      :       scipy.stats.geom(1/138.),
            'max_depth'         :       scipy.stats.randint(2,7),
            'learning_rate'     :       scipy.stats.expon(0, 0.038),
            'min_samples_leaf'  :       scipy.stats.geom(1/9.),
            'subsample'         :       scipy.stats.beta(17,3)
            }

clf = GradientBoostingClassifier()

best_gb, best_hyperparams, best_gb_auc = GridSearch(
                        classifier      =       clf,
                        paramdict       =       params,
                        iters           =       243,
                        X               =       X_train,
                        y               =       y_train,
                        X_reserve       =       None,
                        y_reserve       =       None
            )

print 'Best GB hyperparams: %s' % best_hyperparams
print 'GB AUC: %s' % best_gb_auc
