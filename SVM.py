from dataloader import import_data
from GridSearch import GridSearch
import numpy as np
import pandas as pd
import scipy
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#===================================prep data==================================

X_train, y_train, X_test, id_test = import_data('train.csv', 'test.csv', 'TARGET', 'ID', verbose=True)

#================================SVM===========================================

params = {'C': scipy.stats.expon(100),
            'gamma': scipy.stats.expon(0.01)}

clf = SVC(probability=True)

best_svm, best_hyperparams, best_svm_auc = GridSearch(
                        classifier      =       clf,
                        paramdict       =       params,
                        iters           =       25,
                        X               =       X_train,
                        y               =       y_train
            )

print 'Best SVM hyperparams: %s' % best_hyperparams
print 'SVM AUC: %s' % best_svm_auc

best_svm.fit(X_train, y_train)
# predicting
svm_pred = best_svm.predict_proba(X_test)[:,1]
