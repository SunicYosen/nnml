#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from bpnn import BPNN2
from load_data import load_data

def main():
    # Load data
    data_file               = "data_train.txt"
    data_array, label_array = load_data(data_file)
    data_mat                = np.mat(data_array)
    label_mat               = label_array

    # To 0-1
    data_mat  = preprocessing.MinMaxScaler().fit_transform(data_mat)
    label_mat = preprocessing.MinMaxScaler().fit_transform(label_mat)

    # Get min max
    max_label = 0
    min_label = 0
    for i in range(len(label_array)):
        if label_mat[i] == 1.0:
            max_label = label_array[i][0]
            print("MAX DATA:", max_label)
        if label_mat[i] == 0.0:
            min_label = label_array[i][0]
            print("MIN DATA:", min_label)

    # Train and test data
    data_train, data_test, label_train, label_test = train_test_split(data_mat, label_mat, test_size = 0.2)

    # Define parameters
    input_n       = np.shape(data_train)[1]
    hidden_n      = 10
    output_n      = 1
    active_method = 'sigmoid'
    learn_rate    = 0.2
    correct_belta = 0.2
    max_iters     = 50

    # Set NN
    bpnn            = BPNN2()
    bpnn.setup(input_n=input_n, hidden_n=hidden_n, output_n=output_n, \
             active_method=active_method, learn_rate=learn_rate, \
             correct_belta=correct_belta, max_iters=max_iters)
    # Train
    errors = bpnn.train(data_train, label_train)
    print("Train ERROR = ", errors[-1])

    # Test
    test_results = bpnn.predict(data_test)
    test_error   = mean_squared_error(test_results, label_test)
    print ("Test ERROR = ", test_error)

    ## Get Real result
    real_results = []
    for i in range(len(test_results)):
        real_result = test_results[i][0] * (max_label - min_label) + min_label
        real_results.append(real_result)

    # Plot
    colors   = ['r','g','b','k','c','m','y']
    markers  = ['o','s','p','*','+','x','D','d','v','^','<','>']

    ## plot data result
    plot_trainx = []
    plot_trainy = []
    plot_trainz = []

    for i in range(len(label_train)):
        plot_trainx.append(data_train[i,0])
        plot_trainy.append(data_train[i,1])
        real_label = label_train[i,0] * (max_label - min_label) + min_label
        plot_trainz.append(real_label)

    plot_testx = []
    plot_testy = []
    plot_testz = []
    for i in range(len(real_results)):
        plot_testx.append(data_test[i,0])
        plot_testy.append(data_test[i,1])
        plot_testz.append(real_results[i])

    x = np.random.random([10000])
    y = np.random.random([10000])
    z = np.zeros([10000])
    for i in range(len(x)):
        result = bpnn.predict([np.array((x[i], y[i]))])
        z[i]   = result[0][0] * (max_label - min_label) + min_label
    
    figure = plt.figure("Result")
    fig_plot = figure.add_subplot(111, projection='3d')
    fig_plot.scatter(plot_trainx, plot_trainy, plot_trainz, s=10, marker=markers[0], c=colors[0], label="Train Data") # plot 0
    fig_plot.scatter(plot_testx, plot_testy, plot_testz, s=30, marker=markers[1], c=colors[1], label="Test Data") # plot 0
    fig_plot.plot(x,y,z)

    fig_plot.set_title("The Result Linear Regression")
    fig_plot.set_xlabel('Area')
    fig_plot.set_ylabel('Rooms')
    fig_plot.set_zlabel('Price')
    fig_plot.legend()

    ## Plot Error
    e_fig  = plt.figure("Train Error Curve")
    e_plot = e_fig.add_subplot(111)

    iters = np.shape(errors)[0]
    epoch = np.arange(0, iters, 1)

    e_plot.plot(epoch, errors)
    plt.title("Train Error Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    
    plt.show()

if __name__=='__main__':
    main()
