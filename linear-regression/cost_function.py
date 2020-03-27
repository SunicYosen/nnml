# /usr/bin/python3

from functions import h_theta

def cost_function(THETA, X, Y):
    H_THETA       = h_theta(THETA, X)
    DELTA         = H_THETA - Y
    cost_total    = DELTA.T * DELTA
    return cost_total / 2.0
