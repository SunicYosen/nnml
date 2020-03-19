#
# Sigmoid Function
# x: input data array 
# 

import numpy as np

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z)) # Return sigmoid result

def gradient(THETA, X, Y):
    h_theta = sigmoid(X * THETA)
    result  = X.T * (Y - h_theta)
    return result

def hessian_matrix(THETA, X, Y):
    m, n    = np.shape(X)
    h_theta = sigmoid(X * THETA)
    temp    = np.multiply(h_theta, (1-h_theta))
    M       = X / m
    for i in range(n):
        M[:,i] = np.multiply(M[:,i], temp)

    result  = X.T * M

    return result

def cost_func(THETA, X, Y):
    m, n    = np.shape(X)
    h_theta = sigmoid(X * THETA)
    cost_j  = (-Y.T * np.log(h_theta) - (1-Y).T * np.log(1-h_theta)) / m
    return cost_j
