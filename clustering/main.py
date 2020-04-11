#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from load_data import load_data
from clustering import clustering
from plot import plot_scatter_2d
import matplotlib.pyplot as plt

def main():
    k = 2
    data_file  = "data.txt"
    data_array = load_data(data_file)
    data_mat   = np.mat(data_array)

    cluster    = clustering(k=k, method='k_mediods')
    cluster.fit(data_mat)

    centroids  = cluster._centroids
    labels     = cluster._labels
    error      = cluster._error

    print(error)

    # Plot The Result

    colors   = ['r','g','b','k','c','m','y','#e24fff','#524C90','#845868']
    markers  = ['o','s','p','*','+','x','D','d','v','^','<','>',]

    figure   = plt.figure("Result")
    fig_plot = figure.add_subplot(111)

    for i in range(k):
        cluster_i_index = np.nonzero(labels == i)
        cluster_i_data1 = np.asarray(data_mat[cluster_i_index][:,0])
        cluster_i_data2 = np.asarray(data_mat[cluster_i_index][:,1])

        fig_plot.scatter(cluster_i_data1, cluster_i_data2, marker=markers[i], c=colors[i], s=20)
        fig_plot.scatter(centroids[i,0], centroids[i,1], marker='x', c=colors[i], s=40)
        fig_plot.text(centroids[i,0], centroids[i,1], 'G', c=colors[i], fontdict={'weight': 'bold', 'size': 14})

    plt.title("The Result of Cluster")
    plt.xlabel("Featrue 1")
    plt.ylabel("Featrue 2")

    # Plot Error
    sse_fig  = plt.figure("SSE Curve")
    sse_plot = sse_fig.add_subplot(111)
    
    iters = np.shape(error)[0]
    epoch = np.arange(0, iters, 1)

    sse_plot.plot(epoch, error)
    plt.title("SSE Curve")
    plt.xlabel('Epoch')
    plt.ylabel('SSE')
    
    plt.show()


if __name__=='__main__':
    main()
