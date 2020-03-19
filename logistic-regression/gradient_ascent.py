#
# Gradient Ascent
#
import numpy as np
from functions import sigmoid
from functions import gradient
from functions import cost_func

#
# Gradient Ascent Function
# data_mat: (m,n) m-data_numbers, n-variable_num
# label_mat: (1, m) m-data_numbers
#
def gradient_ascent(data_array, label_array, step_alpha, max_iter_numbers):
    data_matrix  = np.mat(data_array)
    label_matrix = np.mat(label_array) #

    m,n         = np.shape(data_matrix)
    
    theta       = np.zeros((n,1))    # Initial theta
    alpha       = step_alpha         # Step for gradient ascent
    epoches     = max_iter_numbers  # Times for Iteration
    cost_vector = []

    for i in range(epoches):
        cost_j           = cost_func(theta, data_matrix, label_matrix)
        cost_vector.append(cost_j[0,0])
        theta            = theta + alpha * gradient(theta, data_matrix, label_matrix)
        # Gradient Ascent
    
    cost_j               = cost_func(theta, data_matrix, label_matrix)
    cost_vector.append(cost_j[0,0])

    return theta, cost_vector

