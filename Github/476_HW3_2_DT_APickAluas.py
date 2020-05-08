#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:56:31 2020

@author: alex
"""

import numpy as np
from sklearn import svm, tree, metrics

data = np.load('adult-data.npy')
labels = np.load('adult-labels.npy')

# Splitting data into training and testing portions
# and splitting data into parameters and labels
i = int(.9*data.shape[0])
train_data = data[:i]
train_labels = labels[:i]
test_data = data[i:]
test_labels = labels[i:]

# SVM was hitting about 74% accuracy and took over a minute to run
#svc = svm.SVC(kernel = 'sigmoid', gamma = 'auto').fit(train_data, train_labels)
#test_pred = svc.predict(test_data)

# Training and testing portion of Decision Tree classifier
dtc = tree.DecisionTreeClassifier().fit(train_data, train_labels)
test_pred = dtc.predict(test_data)

print("Testing accuracy: " + str(metrics.accuracy_score(test_pred, test_labels)))