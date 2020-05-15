#
# Main
#

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from load_data import load_train_data, load_test_data
from test import test
from train import train

def main():
    train_file              = "data_train.txt"
    test_file               = "data_test.txt"
    epoches                 = 100
    alpha                   = 0.000000001
    data_array, label_array = load_train_data(train_file)
    test_array              = load_test_data(test_file)
    data_matrix             = np.mat(data_array)
    label_matrix            = np.mat(label_array)
    test_matrix             = np.mat(test_array)
    theta, cost_vector      = train(data_matrix, label_matrix, epoches, alpha)
    test_result             = test(theta, test_matrix)
    print(theta)
    print(cost_vector[np.size(cost_vector)-1])
    print(test_matrix, test_result)

    # Plot Result
    m,n       = np.shape(data_array)
    plot_x = []
    plot_y = []
    plot_z = []

    for i in range(m):
        plot_x.append(data_matrix[i,1])
        plot_y.append(data_matrix[i,n-1])
        plot_z.append(label_matrix[i,0])

    test_m, test_n       = np.shape(test_matrix)
    plot_testx = []
    plot_testy = []
    plot_testz = []
    for i in range(test_m):
        plot_testx.append(test_matrix[i,1])
        plot_testy.append(test_matrix[i,test_n-1])
        plot_testz.append(test_result[i,0])
    
    figure = plt.figure("Result")
    fig_plot = figure.add_subplot(111, projection='3d')
    fig_plot.scatter(plot_x, plot_y, plot_z, s=5, c='red', marker='s') # plot 0
    fig_plot.scatter(plot_testx, plot_testy, plot_testz, s=30, c='green', marker='s') # plot 0
    x = np.random.randint(1000, 5000, size=[10000])
    y = np.random.randint(2, 5, size=[10000])
    z = theta[0,0] + theta[1,0] * x + theta[2,0] * y
    fig_plot.plot(x,y,z)
    fig_plot.set_title("The Result Linear Regression")
    fig_plot.set_xlabel('Area')
    fig_plot.set_ylabel('Rooms')
    fig_plot.set_zlabel('Price')

    # Plot Cost
    cost_fig  = plt.figure("Cost")
    cost_plot = cost_fig.add_subplot(111)
    epoch   = np.arange(0, epoches+1, 1)
    cost_plot.plot(epoch, cost_vector)
    plt.title("The Cost")
    plt.xlabel('Epoch')
    plt.ylabel('Cost')

    plt.show()

if __name__=="__main__":
    main()

