#
# Sigmoid Function
# x: input data array 
# 

import numpy as np

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z)) # Return sigmoid result

def gradient(H_THETA, X, Y):
    m = np.shape(Y)[0]
    result = (H_THETA - Y).T * X / m
    return result

def hessian_matrix(H_THETA, X, Y):
    m = np.shape(Y)[0]
    temp = np.multiply(H_THETA, (1-H_THETA))
    M    = X / m
    for i in range(np.shape(X)[1]):
        M[:,i] = np.multiply(M[:,i], temp)

    return X.T * M




