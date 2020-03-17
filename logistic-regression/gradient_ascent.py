#
# Gradient Ascent
#
import numpy as np
from functions import sigmoid

#
# Gradient Ascent Function
# data_mat: (m,n) m-data_numbers, n-variable_num
# label_mat: (1, m) m-data_numbers
#
def gradient_ascent(data_array, label_array):
    data_matrix  = np.mat(data_array, dtype=np.float128)
    label_matrix = np.mat(label_array, dtype=np.float128) #

    m,n     = np.shape(data_matrix)

    weights = np.zeros((n,1), dtype=np.float128)    # Initial weights
    alpha   = 0.00001         # Step for gradient ascent
    epoches = 200000        # Times for Iteration

    for i in range(epoches):
        h       = sigmoid(data_matrix * weights)
        error   = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error

    return weights

