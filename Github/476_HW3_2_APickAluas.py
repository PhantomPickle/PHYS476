#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:56:31 2020

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm

data = np.load('adult-data.npy')
labels = np.load('adult-labels.npy')

#train_data = data[]

# Defining SVM parameters: C = regularization parameter, 
C = 1.0
svc = svm.SVC(kernel = 'linear', C = 1, gamma = 0.5).fit(data, labels)
#pred = svc.predict()

print(data.shape)