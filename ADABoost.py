from dataloader import import_data
from GridSearch import GridSearch
import numpy as np
import pandas as pd
import scipy
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

#===================================prep data==================================

np.random.seed(42)

target_col = 'TARGET'
id_col = 'ID'

X_train = pd.read_csv('./EngineeredData/Xtrain.csv')
y_train = pd.read_csv('./EngineeredData/ytrain.csv')[target_col]
X_test = pd.read_csv('./EngineeredData/Xtest.csv')
id_test = pd.read_csv('./EngineeredData/idtest.csv')[id_col]

#=============================ADA Boster========================================

params = {
            'n_estimators'        :       scipy.stats.geom(1/547.),
            'learning_rate'       :       scipy.stats.expon(0, 0.0264)
            }


clf = AdaBoostClassifier()

GridSearch(
                        classifier      =       clf,
                        paramdict       =       params,
                        iters           =       9,
                        X               =       X_train,
                        y               =       y_train,
                        X_reserve       =       None,
                        y_reserve       =       None
)
