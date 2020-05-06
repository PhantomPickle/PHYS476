#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:38:21 2020

@author: alex
"""

import h5py
import numpy as np


file = h5py.File('Galaxy10.h5', 'r')
ans_data = file['ans']
img_data = file['images']

print(ans_data.shape)