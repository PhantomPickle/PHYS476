#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:00:00 2020

@author: alex
"""

import numpy as np
import pandas as pd
from sklearn import svm, tree, metrics

data = pd.read_csv('breast-cancer-wisconsin.data', na_values = ['?'])
data = data.dropna()

train_data = data.sample(frac = 0.9, axis = 0)
test_data = data.drop(train_data.index)
del data

train_labels = train_data[train_data.columns[-1]].to_numpy()
train_data = train_data.drop(train_data.columns[-1], axis = 1).to_numpy()

test_labels = test_data[test_data.columns[-1]].to_numpy()
test_data = test_data.drop(test_data.columns[-1], axis = 1).to_numpy()

svc = svm.SVC(kernel = 'poly', gamma = 'auto').fit(train_data, train_labels)
test_pred = svc.predict(test_data)

#dtc = tree.DecisionTreeClassifier().fit(train_data, train_labels)
#test_pred = dtc.predict(test_data)

print("Testing accuracy: " + str(metrics.accuracy_score(test_pred, test_labels)))