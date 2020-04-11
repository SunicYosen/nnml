#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math

from distance import *
from functions import rand_center, find_distance_min
from k_means import k_means_update_centroids
from k_mediods import k_mediods_update_centroids

class clustering():
    def __init__(self, k=3, init_center='random', method='k_means', max_iter = 500, conver_toler = 0, distance_method = ('euclidean', 2)):
        '''
        Function: Init K_means
        '''
        self._k               = k
        self._init_center     = init_center
        self._method          = method
        self._max_iter        = max_iter
        self._conver_toler    = conver_toler
        self._distance_method = distance_method
        self._centroids       = None
        self._cluster_assment = None
        self._labels          = None
        self._error           = []

    def fit(self, data_mat):
        '''
        Function: K-MEANS training
        Input: data mat
        '''

        if not isinstance(data_mat, np.matrixlib.defmatrix.matrix):
            try:
                data_mat = np.mat(data_mat)
            except:
                raise TypeError("Data_mat should be np.mat!")

        m,n = np.shape(data_mat)
        self._cluster_assment = np.zeros((m,2))
        # The first col is THE Cluster
        # THe second col is the error/lose with centroids

        if self._init_center == 'random':
            self._centroids = rand_center(data_mat, self._k)

        print(self._centroids)

        cluster_changed_flag = True
        epoch                = 0

        while (epoch < self._max_iter) and (cluster_changed_flag):
            cluster_changed_flag = False

            for i in range(m):
                min_centroids_index, min_distance = find_distance_min(data_mat[i,:], self._centroids, distance_method=self._distance_method)

                if (self._cluster_assment[i, 0] != min_centroids_index) or ((self._cluster_assment[i,1]) >  min_distance**(self._distance_method[1])):
                    cluster_changed_flag = True
                
                self._cluster_assment[i,:] = min_centroids_index, min_distance**(self._distance_method[1])  # SSR

            if not cluster_changed_flag:
                break

            # Update Centroids
            if self._method == 'k_means':
                new_centroids = k_means_update_centroids(data_mat, self._cluster_assment[:,0], self._k)
            elif self._method == 'k_mediods':
                new_centroids = k_mediods_update_centroids(data_mat, self._cluster_assment[:,0] ,self._k, self._distance_method)

            else:
                raise NameError("No method named: ", self._method)

            self._error.append(sum(self._cluster_assment[:,1]))
            self._centroids = new_centroids

        self._labels  = self._cluster_assment[:,0]

    def predict(self, data_test):
        if not isinstance(data_test, np.matrixlib.defmatrix.matrix):
            try:
                data_test = np.mat(data_test)
            except:
                raise TypeError("Data_test should be np.mat!")

        m,n = np.shape(data_test)
        predicts = np.empty((m,))

        for i in range(m):
            min_distance = float('inf')
            min_centroids_index, min_distance = find_distance_min(data_test[i,:], self._centroids, distance_method=self._distance_method)

            predicts[i] = min_centroids_index

        return predicts
