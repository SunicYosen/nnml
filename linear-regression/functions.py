# /usr/bin/python3

def h_theta(THETA, X):
    result = X * THETA
    return result

def gradient(THETA, X, Y):
    H_THETA        = h_theta(THETA, X)
    gradient_value = X.T * (Y - H_THETA)
    return gradient_value
