#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from GridSearch import GridSearch
import numpy as np
import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


#===================================prep data==================================

np.random.seed(42)

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./Level1Data/Xtrain.csv')
X_train['GBpred'] = pd.read_csv('./Level1Data/GBPredtrain.csv')
X_train['ADApred'] = pd.read_csv('./Level1Data/ADAPredtrain.csv')
y_train = pd.read_csv('./Level1Data/ytrain.csv')[target_col]


#==========================Gradient Boost Classifier===========================

params = {
            'n_estimators'      :       scipy.stats.geom(1/150.),
            'max_depth'         :       scipy.stats.randint(2,7),
            'learning_rate'     :       scipy.stats.expon(0, 0.01),
            'min_samples_leaf'  :       scipy.stats.geom(1/10.),
            'subsample'         :       scipy.stats.beta(2,1),
            'colsample_bytree'  :       scipy.stats.beta(2,1)
            }

clf = XGBClassifier()

GridSearch(
                        classifier      =       clf,
                        paramdict       =       params,
                        iters           =       729,
                        X               =       X_train,
                        y               =       y_train,
                        X_reserve       =       None,
                        y_reserve       =       None
)
