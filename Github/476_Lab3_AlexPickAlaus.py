#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:01:09 2020

@author: alex
"""

import torch
import numpy as np
import torch.nn.functional as func

train_data = np.genfromtxt('mnist_train.csv', delimiter=',')
test_data = np.genfromtxt('mnist_test.csv', delimiter=',')

#train_data = np.load("train_saved.npy")
#test_data = np.load("test_saved.npy")

# Splitting training and testing sets into input and output data;
# Input data mapped from {0-255} to {0-1}
train_in = train_data[:,1:]/255.
train_out = train_data[:,0]
test_in = test_data[:,1:]/255.
test_out = test_data[:,0]
del train_data, test_data

# Ensuring type compatibility 
train_in = torch.from_numpy(np.float32(train_in))
train_out = torch.from_numpy(train_out).long()
test_in = torch.from_numpy(np.float32(test_in))
test_out = torch.from_numpy(test_out).long()

# b_frac = batch fraction of total data, b_num = number of batches
# b_size = batch size, i_dim = input dimension
# h_dim = hidden dimension, o_dim = output dimension
b_frac = .05
b_num = int(1/b_frac)
b_size = int(b_frac*train_in.shape[0])
i_dim = train_in.shape[1]
h_dim = 128
o_dim = 10

# Initializing random weights on edges from input --> hidden (i_w) 
# and hidden --> output (o_w);
# Normalized by sqrt(dimensionality)
i_w = torch.randn(i_dim, h_dim)/np.sqrt(i_dim)
o_w = torch.randn(h_dim, o_dim)/np.sqrt(h_dim)
i_w.requires_grad_()
o_w.requires_grad_()

# Initializing random biases on nodes in hidden and output layers
h_b = torch.rand(h_dim, requires_grad = True)
o_b = torch.rand(o_dim, requires_grad = True)

# Runs data through the network
# clamp() applies ReLU on activations for hidden layer     
def network(in_vals):
    out_vals = in_vals.mm(i_w) + h_b
    out_vals = out_vals.clamp(min=0).mm(o_w) + o_b
    return out_vals

# Computes accuracy of predicted values
def accuracy(test_vals, real_vals):
    max_vals = torch.argmax(test_vals, 1)
    acc = (max_vals == real_vals).float().mean()
    return acc

# lr = learning rate, e_num = number of epochs
lr = .01
e_num = 100

# Training: outer loop iterates over epochs, inner loop iterates over batches
for e in range(e_num):
    for b in range(b_num):
        b_start = b * b_size
        b_end = (b+1) * b_size
        batch_in = train_in[b_start : b_end]
        batch_out = train_out[b_start : b_end]
       
        # o_p = predicted output, weighted and biased, activated,
        # then weighted and biased again
        o_p = network(batch_in)
        train_loss = func.cross_entropy(o_p, batch_out)
        train_acc = accuracy(o_p, batch_out)
        train_loss.backward()
        
        with torch.no_grad():
            i_w -= lr * i_w.grad
            o_w -= lr * o_w.grad
            i_w.grad.zero_()
            o_w.grad.zero_()
            
            h_b -= lr * h_b.grad
            o_b -= lr * o_b.grad
            h_b.grad.zero_()
            o_b.grad.zero_()
            
    print("Epoch: " + str(e+1) + ", Loss: " + str(train_loss.item()) + ", Accuracy: " + str(train_acc.item()))
    
# Testing    
test_pred = network(test_in)
test_loss = func.cross_entropy(test_pred, test_out)
test_acc = accuracy(test_pred, test_out)
print("Testing accuracy: " + str(test_acc.item()))
        

        