#! /usr/bin/python3

import numpy as np 
import matplotlib.pyplot as plt

def plot_scatter_2d(X, Y, figure_name="Result", size=30, color='black', marker='s', label=" ", title=" ", label_x="x", label_y="y"):
    figure   = plt.figure(figure_name)
    fig_plot = figure.add_subplot(111)
    fig_plot.scatter(X, Y, s=size, c=color, marker=marker, label=label)
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    return 0


if __name__ == "__main__":
    x = [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 8.0, 9.0]
    y = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 5.0]
    plot_scatter_2d(x, y)
    plt.show()