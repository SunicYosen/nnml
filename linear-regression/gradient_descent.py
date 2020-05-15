#
# Gradient Descent
#

import numpy as np

from functions import gradient
from cost_function import cost_function

def gradient_descent(data_matrix, label_matrix, alpha, max_iter_numbers):
    step_alpha    = alpha
    epoches       = max_iter_numbers

    m,n           = np.shape(data_matrix)
    theta         = np.zeros((n,1))    # Initial theta
    cost_vector   = []

    for epoch in range(epoches):
        cost_j           = cost_function(theta, data_matrix, label_matrix)
        cost_vector.append(cost_j[0,0])
        theta            = theta + step_alpha * gradient(theta, data_matrix, label_matrix)
    
    cost_j               = cost_function(theta, data_matrix, label_matrix)
    cost_vector.append(cost_j[0,0])

    return theta, cost_vector
