#
# Newton Method
#
import numpy as np
from functions import sigmoid
from functions import gradient
from functions import hessian_matrix

#
# newton method
# data_mat: (m,n) m-data_numbers, n-variable_num
# label_mat: (1, m) m-data_numbers
#

def newton_method(data_array, label_array, max_iter_count = 500):
    data_matrix  = np.mat(data_array)
    label_matrix = np.mat(label_array) #

    (m, n)  = np.shape(data_matrix)

    weights = np.zeros((n,1), dtype=np.float128)    # Initial weights
    
    alpha   = 0.00001               # Step for gradient ascent
    epoches = max_iter_count        # Times for Iteration

    for i in range(epoches):
        h        = sigmoid(data_matrix * weights)
        error    = label_matrix - h
        g        = gradient(h, data_matrix, label_matrix)
        hessian_m= hessian_matrix(h, data_matrix, label_matrix)

        weights  = weights - (hessian_m.I * g.T)

    return weights

