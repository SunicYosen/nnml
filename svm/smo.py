#! /usr/bin/python3

import numpy as np 
from svm_struct import svm_struct
from functions import *

def smo(data_mat, label_mat, C, toler, max_iter, kernel_type=('linear', 0)):
    svm_s = svm_struct(data_mat, label_mat, C, toler, kernel_type)
    epoch = 0 # iter parameter
    entire_set = True
    alpha_pairs_changed = 0

    while (epoch < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(svm_s.m): # For each data
                alpha_pairs_changed += inner_l(i, svm_s)
                # print("Fullset, epoch:%d i:%d, pairs changed %d" % (epoch, i, alpha_pairs_changed))
            epoch += 1
        else:
            non_bound_is = np.nonzero((svm_s.alphas.A > 0) * (svm_s.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, svm_s)
                # print("Non-bound, epoch: %d i: %d , pairs changed %d" % (epoch, i , alpha_pairs_changed))

            epoch += 1

        if entire_set:
            entire_set = False

        elif (alpha_pairs_changed == 0):
            entire_set = True

        print("Iteration Number: %d" % epoch)

    return svm_s.b, svm_s.alphas
            