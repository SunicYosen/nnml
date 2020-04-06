#! /usr/bin/python3

import numpy as np 
import math

# Linear kernel
# X: Input Data, Suport Vector

def linear_kernel(X, A):
    m,n    = np.shape(X)
    kernel = np.mat(np.zeros((m,1)))
    kernel = X * (A.T)
    return kernel

def rbf_kernel(X, A, param):

    m,n    = np.shape(X)
    kernel = np.mat(np.zeros((m,1)))
    for i in range(m):
        delta_row = X[i,:] - A
        kernel[i] = delta_row * delta_row.T

    kernel = np.exp(kernel / (-2.0 * (param ** 2)))
    
    '''
    if isinstance(param, float) == False:
        param = 0.5
    
    kernel = math.exp(-np.linalg.norm(X - A)**2 / (2 * param**2))
    '''

    return kernel

def kernel(X, A, kernel_type):
    m,n    = np.shape(X)
    kernel = np.mat(np.zeros((m, 1)))
    if 'linear' in kernel_type:
        kernel = linear_kernel(X, A)
    elif 'rbf' in kernel_type:
        param  = kernel_type[1]
        kernel = rbf_kernel(X, A, param)

    else:
        raise NameError("That Kernel is not defined!")

    return kernel
