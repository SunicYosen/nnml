#! /usr/bin/python3

import numpy as np
from load_data import load_data
from smo import smo
from kernel import kernel

def train(data_mat, label_mat, C=2.0, toler=0.01, max_iter=100, kernel_type=('linear', 0)):
    b, alphas = smo(data_mat, label_mat, C, toler, max_iter, kernel_type)
    support_vector_index  = np.nonzero(alphas)[0]
    support_vector_data   = data_mat[support_vector_index]
    support_vector_label  = label_mat[support_vector_index]
    support_vector_alphas = alphas[support_vector_index]

    print("There are %d support vectors" % np.shape(support_vector_data)[0])
    m,n = np.shape(data_mat)
    error_count = 0

    for i in range(m):
        kernel_eval = kernel(support_vector_data, data_mat[i,:], kernel_type)
        predict     = kernel_eval.T * np.multiply(support_vector_label, support_vector_alphas) + b

        if np.sign(predict) != np.sign(label_mat[i,0]):
            error_count += 1

    print("The Training error rate is: %d/%d" % (error_count, m))

    return support_vector_data, support_vector_label, support_vector_alphas, b