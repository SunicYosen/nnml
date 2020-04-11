#! /usr/bin/python3

import numpy as np
from functions import distances_sum

def find_centroid(data_cluster_mat, distance_method):

    min_distances = float('inf')
    m,n           = np.shape(data_cluster_mat)
    centroid      = np.zeros((n,))

    for controid_temp in data_cluster_mat:
        distances = distances_sum(controid_temp, data_cluster_mat, distance_method=distance_method)
        if distances < min_distances:
            min_distances = distances
            centroid      = controid_temp

    return centroid



def k_mediods_update_centroids(data_mat, data_cluster, k, distance_method):
    '''
    Function: update centroids with k_mediods method
    Input: Data_mat and data_cluster
    '''
    m,n             = np.shape(data_mat)
    centroids_new   = np.empty((k, n))

    #
    for i in range(k):
        data_cluster_index = np.nonzero(data_cluster == i)
        data_cluster_mat   = data_mat[data_cluster_index]
        new_centroid       = find_centroid(data_cluster_mat, distance_method)
        centroids_new[i,:] = new_centroid

    return centroids_new


if __name__=='__main__':
    data_mat = [[0,0],[1,1],[10,10],[9,9]]
    data_cluster = [[0],[0],[1],[1]]
    k = 2

    centroids = k_mediods_update_centroids(data_mat, data_cluster, k)

    print(centroids)