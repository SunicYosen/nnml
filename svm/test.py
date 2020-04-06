#! /usr/bin/python3

import numpy as np
from load_data import load_data
from smo import smo
from kernel import kernel

def test(data_mat, label_mat_ref, support_vector_data, support_vector_label, support_vector_alphas, b, kernel_type):
    error_count = 0
    m, n        = np.shape(data_mat)
    for i in range(m):
        kernel_eval = kernel(support_vector_data, data_mat[i,:], kernel_type)
        predict     = kernel_eval.T * np.multiply(support_vector_label, support_vector_alphas) + b
        if np.sign(predict) != np.sign(label_mat_ref[i]):
            error_count += 1
            print(i)

    print("The Test error rate is: %d/%d" % (error_count, m))
    return m, error_count
