#! /usr/bin/python3
import numpy as np

from kernel import kernel

class svm_struct:
    def __init__(self, data_mat, label_mat, C, toler, kernel_type):
        self.data_mat  = data_mat                   # Data Mat
        self.label_mat = label_mat                  # Label Mat
        self.C         = C                          # Soft margin parameter
        self.toler     = toler                      # toler for stop
        self.m         = np.shape(data_mat)[0]      # numbers of data mat
        self.alphas    = np.mat(np.zeros((self.m, 1))) # alphas mat
        self.b         = 0                          # Initial as 0
        self.eCache    = np.mat(np.zeros((self.m, 2))) # Cache
        self.Kernel    = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.Kernel[:, i] = kernel(self.data_mat, self.data_mat[i, :], kernel_type)
