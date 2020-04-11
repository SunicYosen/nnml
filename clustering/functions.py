#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import datetime

import distance

def rand_center(data_mat, k):
    '''
    Function: initial center point randomly
    Input: Data_mat and k
    '''
    random.seed(datetime.datetime.now())

    m,n         = np.shape(data_mat)
    centroids   = np.empty((k, n))
    random_list = random.sample(range(m),k)

    for i in range(k):
        centroids[i] = data_mat[random_list[i]]

    return centroids

def find_distance_min(point, point_array, distance_method = ('euclidean', 2)):
    min_distance = float('inf')
    min_centroid_index = -1

    k = np.shape(point_array)[0]

    point = np.asarray(point)[0]
    
    for j in range(k):
        centroid_j = point_array[j]
        if distance_method[0] == 'euclidean':
            distance_j = distance.euclidean_distance(point, centroid_j)
        elif distance_method[0] == 'manhattan':
            distance_j = distance.manhattan_distance(point, centroid_j)
        elif distance_method[0] == 'minkowski':
            distance_j = distance.minkowski_distance(point, centroid_j, distance_method[1])
        else:
            distance_j = distance.euclidean_distance(point, centroid_j)
        '''or
        distance_j = distance.minkowski_distance(point, centroid_j, distance_method[1])
        '''

        if distance_j < min_distance:
            min_distance = distance_j
            min_centroid_index = j
    
    return min_centroid_index, min_distance


def distances_sum(point, point_array, distance_method = ('euclidean', 2)):
    distances = float(0)

    m = np.shape(point_array)[0]
    point = np.asarray(point)[0]
    
    for j in range(m):
        if distance_method[0] == 'euclidean':
            distances += np.power(distance.euclidean_distance(point, np.asarray(point_array[j])[0]), distance_method[1])
        elif distance_method[0] == 'manhattan':
            distances += np.power(distance.manhattan_distance(point, np.asarray(point_array[j])[0]), distance_method[1])
        elif distance_method[0] == 'minkowski':
            distances += np.power(distance.minkowski_distance(point, np.asarray(point_array[j])[0], distance_method[1]), distance_method[1])
        else:
            distances += np.power(distance.euclidean_distance(point, point_array[j]), distance_method[1])
        '''or
        distance_j = distance.minkowski_distance(point, centroid_j, distance_method[1])
        '''
    
    return distances
