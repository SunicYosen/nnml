#
# Train
#

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import gradient_descent
from normal_equations import normal_quations
from load_data import load_train_data, load_test_data
from test import test

def train(data_mat, label_mat, epoches, alpha):
    max_iter_count          = epoches
    step_alpha              = alpha
    theta, cost_v           = gradient_descent(data_mat, label_mat, step_alpha, max_iter_count)
    # theta, cost_v           = normal_quations(data_mat, label_mat)
    return theta, cost_v

