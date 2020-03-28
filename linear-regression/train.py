#
# Train
#

from gradient_descent import gradient_descent
from normal_equations import normal_quations

def train(data_mat, label_mat, epoches, alpha):
    max_iter_count          = epoches
    step_alpha              = alpha
    theta, cost_v           = gradient_descent(data_mat, label_mat, step_alpha, max_iter_count)
    # theta, cost_v           = normal_quations(data_mat, label_mat)
    return theta, cost_v

