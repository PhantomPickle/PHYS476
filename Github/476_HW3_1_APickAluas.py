#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:38:21 2020

@author: alex
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

seed = 7
np.random.seed(seed)
torch.manual_seed(seed)

# Designating GPU usage
device = torch.device("cuda:0")

# Loading dataset
file = h5py.File('Galaxy10.h5', 'r')
img_data = file['images']
ans_data = file['ans']

# Rearranging image dimensions to be compatible with PyTorch
img_data = np.moveaxis(img_data, -1, 1)

# Trimming dataset to a more manageable size
s_img_data = img_data[:50000]
s_ans_data = ans_data[:50000]
del img_data
del ans_data

# Splitting into training and testing sets
train_frac = .9
f_index = int(train_frac * len(s_img_data))
train_in = s_img_data[:f_index]
train_out = s_ans_data[:f_index]
test_in = s_img_data[f_index:]
test_out = s_ans_data[f_index:]
del s_img_data
del s_ans_data

train_in = train_in / 255.
test_in = test_in / 255.

# Ensuring type compatibility 
train_in = torch.from_numpy(np.float32(train_in))
train_out = torch.from_numpy(train_out).long()
test_in = torch.from_numpy(np.float32(test_in))
test_out = torch.from_numpy(test_out).long()

# Network hyperparameters
learn_rate = .001
b_frac = .05
epochs = 10

batches = int(1/b_frac)
b_size = int(b_frac*train_in.shape[0])

# Defines a CNN class inheriting from the Module base class
class Galaxy_Net(nn.Module):
    def __init__(self):
        super(Galaxy_Net, self).__init__()
        self.c1 = nn.Conv2d(3, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.c_drop = nn.Dropout2d(.4)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        # Pooled relu activated output from first convolutional layer
        x = self.pool(F.relu(self.c1(x)))
        # Dropout of 25% on second convolutional layer
        x = self.c_drop(self.c2(x))
        # Pooled relu activated output from second convolutional layer
        x = self.pool(F.relu(x))
        # Determining input dim for first fully connected layer
        x = x.view(b_size, 16 * 14 * 14)
        # Relu activated output from first fully connected layer
        x = F.relu(self.fc1(x))
        # Activationless output from second fully connected layer
        x = self.fc2(x)
        return x


# Computes accuracy of network outputs (relative to actual labels)
def accuracy(est_labels, labels):
    max_vals = torch.argmax(est_labels, 1)
    acc = (max_vals == labels).float().mean()
    return acc

# Instantiating the network and defining the optimizer
net = Galaxy_Net()
net.to(device)
loss = nn.CrossEntropyLoss()
opti = opt.Adam(net.parameters(), lr = learn_rate)

# Data iteration loop for training
for e in range(epochs):
    for b in range(batches):
        b_start = b * b_size
        b_end = (b+1) * b_size
        batch_in = train_in[b_start : b_end].to(device)
        batch_out = train_out[b_start : b_end].to(device)

        # Zeroes out gradient parameters 
        opti.zero_grad() 
        
        # Predicted output as determined by current network state
        train_pred = net(batch_in)
        
        # Computes loss and accuracy of network predictions with respect to actual labels
        train_loss = loss(train_pred, batch_out)
        train_acc = accuracy(train_pred, batch_out)
        
        # Back propagation of gradient
        train_loss.backward()
        
        # Adjusts weights
        opti.step()
        
    print("Epoch: " + str(e+1) + ", Loss: " + str(train_loss.item()) + ", Accuracy: " + str(train_acc.item()))

# Testing iteration loop
test_accs = []
#batch size recalculated for test data
b_size = int(b_frac*test_in.shape[0])
for b in range(batches):
        b_start = b * b_size
        b_end = (b+1) * b_size
        batch_in = test_in[b_start : b_end].to(device)
        batch_out = test_out[b_start : b_end].to(device)  
        
        test_pred = net(batch_in)
        #test_loss = loss(test_pred, batch_out)
        test_accs.append(accuracy(test_pred, batch_out).item())

print("Testing accuracy: " + str(sum(test_accs)/batches))

#
# Do something with the loss!!!
#

