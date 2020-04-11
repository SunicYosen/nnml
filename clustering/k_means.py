#! /usr/bin/python3

import numpy as np

def k_means_update_centroids(data_mat, data_cluster, k):
    '''
    Function: update centroids with k_means method
    Input: Data_mat and data_cluster
    '''
    m,n         = np.shape(data_mat)
    centroids   = np.empty((k, n))

    data_cluster = np.asarray(data_cluster)

    for i in range(k):
        index_value_i    = np.nonzero(data_cluster == i)
        points_cluster_i = data_mat[index_value_i[0]]
        centroids[i,:]   = np.mean(points_cluster_i, axis=0)

    return centroids


if __name__=='__main__':
    data_mat = [[0,0],[1,1],[10,10],[9,9]]
    data_cluster = [0,0,1,1]
    k = 2

    centroids = k_means_update_centroids(data_mat, data_cluster, k)

    print(centroids)