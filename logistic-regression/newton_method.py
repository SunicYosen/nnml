#
# Newton Method
#
import numpy as np
from functions import sigmoid
from functions import gradient
from functions import hessian_matrix
from functions import cost_func

#
# newton method
# data_mat: (m,n) m-data_numbers, n-variable_num
# label_mat: (1, m) m-data_numbers
#

def newton_method(data_array, label_array, max_iter_count):
    data_matrix  = np.mat(data_array)
    label_matrix = np.mat(label_array) #

    (m, n)  = np.shape(data_matrix)

    theta = np.zeros((n,1))    # Initial theta
    
    epoches = max_iter_count   # Times for Iteration

    cost_vector = []

    for i in range(epoches):
        gradient_v       = gradient(theta, data_matrix, label_matrix)
        hessian_mat      = hessian_matrix(theta, data_matrix, label_matrix)
        cost_j           = cost_func(theta, data_matrix, label_matrix)
        cost_vector.append(cost_j[0,0])
        theta            = theta + (hessian_mat.I * gradient_v/m)
    
    cost_j           = cost_func(theta, data_matrix, label_matrix)
    cost_vector.append(cost_j[0,0])

    return theta, cost_vector

