#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math

def euclidean_distance(point_a, point_b):
    '''
    Function: return the euclidean distance of two points without weights
    Input: Two points
    '''
    distance = math.sqrt(sum(np.power(point_a - point_b, 2)))

    return distance

def manhattan_distance(point_a, point_b):
    '''
    Function: return the manhattan distance of two points without weights
    Input: Two points
    '''
    distance = sum(np.abs(point_a - point_b))

    return distance


def minkowski_distance(point_a, point_b, q):
    '''
    Function: return the minkowski distance of two points without weights
    Input: Two points and q
    '''
    distance = np.math.power(sum(np.power(np.abs(point_a - point_b), q), float(1.0/q)))

    return distance
