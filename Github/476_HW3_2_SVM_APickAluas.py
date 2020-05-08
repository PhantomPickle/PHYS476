#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:00:00 2020

@author: alex
"""

import numpy as np
import pandas as pd
from sklearn import svm, tree, metrics

# Reading data in and eliminating rows with missing data
data = pd.read_csv('breast-cancer-wisconsin.data', na_values = ['?'])
data = data.dropna()

# Splitting data into training and testing portions
train_data = data.sample(frac = 0.9, axis = 0)
test_data = data.drop(train_data.index)
del data

# Splitting datasets into parameters and labels
train_labels = train_data[train_data.columns[-1]].to_numpy()
train_data = train_data.drop(train_data.columns[-1], axis = 1).to_numpy()
test_labels = test_data[test_data.columns[-1]].to_numpy()
test_data = test_data.drop(test_data.columns[-1], axis = 1).to_numpy()

# Training and testing of SVM;
# Using linear or poly kernel caused the fitting to never finish.
svc = svm.SVC(kernel = 'sigmoid', gamma = 'auto').fit(train_data, train_labels)
test_pred = svc.predict(test_data)

# Decision tree consistenly hit ~95% accuracy, decided to use SVM for this one
# because too slow using SVM for the income classifier
#dtc = tree.DecisionTreeClassifier().fit(train_data, train_labels)
#test_pred = dtc.predict(test_data)

print("Testing accuracy: " + str(metrics.accuracy_score(test_pred, test_labels)))