#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def load_data(train_file, test_file):
    '''
    Takes as input:
        train_file: the training csv data filename
        train_file: the test csv data filename
    What it does:
        makes use of the pandas read_csv method to convert the csv data to
        pandas dataframes--one for the training data ad one for the test data-- then returns them
        '''
    return pd.read_csv(train_file), pd.read_csv(test_file)

def zap_empties(train, test):
    '''
    Takes as input:
        train: the training factor data in the form a pandas DataFrame
        test: the test factor data in the form a pandas DataFrame
    What it does:
        finds all columns in the training factor data where the standard
        deviation is zero. This implies those columns are constant in the
        training factor data and therefore cannot be used for learning. The
        columns are removed from both the training and test factor data, which
        are returned
        '''
    sd = train.std()
    empties = sd[sd==0.0].index
    return train.drop(empties,1), test.drop(empties,1)

def zap_dependencies(train, test):
    '''
    Takes as input:
        train: the training factor data in the form a pandas DataFrame
        test: the test factor data in the form a pandas DataFrame
    What it does:
        Converts the training factor data to a numpy matrix then performs a QR
        decomposition on it. Dependent columns are identified as all those that
        do not have pivots (i.e., are within a certain tolerence of zero where
        the pivot should be). These columns are then removed from the training and test factor data, which are returned
    '''
    dependencies = []
    feature_cols = train.columns
    Q, R = np.linalg.qr(np.matrix(train))
    indep_locs = np.where(abs(R.diagonal())>1e-7)[1]
    for i, col in enumerate(feature_cols):
        if i not in indep_locs:
            dependencies.append(col)
            if verbose:
                print col
    return train.drop(dependencies,1), test.drop(dependencies,1)

def import_data(train_file, test_file, target_col, id_col, verbose=False):
    '''
    Takes as input:
        train: the training data in the form a pandas DataFrame
        test: the test data in the form a pandas DataFrame
        target_col: the name of the target (output) column
        id_col: the name of the ID column
    What it does:
        Calls upon helper functions to read in and manipulate data. Splits
        the train and test data into X and y groupings and returns them.
    '''
    df_train, df_test = load_data(train_file, test_file)
    feature_cols = df_train.columns.difference([target_col, id_col])
    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_test, id_test = df_test[feature_cols], df_test[id_col]
    X_train, X_test = zap_empties(X_train, X_test)
    X_train, X_test = zap_dependencies(X_train, X_test, verbose)
    return X_train, y_train, X_test, id_test
