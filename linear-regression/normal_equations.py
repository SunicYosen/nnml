import numpy as np

from functions import gradient
from cost_function import cost_function

def normal_quations(data_matrix, label_matrix):
    cost_vector   = []

    theta            = (data_matrix.T * data_matrix).I * data_matrix.T * label_matrix
    
    cost_j           = cost_function(theta, data_matrix, label_matrix)
    cost_vector.append(cost_j[0,0])

    return theta, cost_vector
