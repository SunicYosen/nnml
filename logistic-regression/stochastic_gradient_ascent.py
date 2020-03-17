#
# Stochastic Gradient Ascent
#
import numpy as np
import random
from functions import sigmoid

#
# Stochastic Gradient Ascent
# data_mat: (m,n) m-data_numbers, n-variable_num
# label_mat: (1, m) m-data_numbers
#
def stochastic_gradient_ascent(data_array, label_array):
    data_matrix  = np.mat(data_array, dtype=np.float128)
    label_matrix = np.mat(label_array, dtype=np.float128) #

    m,n     = np.shape(data_matrix)
    weights = np.zeros((n,1))        # Initial weights
    alpha   = 0.03                 # Step for gradient ascent
    epoches = 200000

    for i in range(epoches):
        r_index = random.randint(0,m-1)
        h       = sigmoid(data_matrix[r_index] * weights)

        error   = label_matrix[r_index] - h
        weights  = weights + alpha * error[0,0] * data_matrix[r_index].transpose()

    return weights

