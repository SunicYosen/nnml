# 
# Score classify
# 

import numpy as np 
import matplotlib.pyplot as plt

from load_data import load_data
from functions import sigmoid
from gradient_ascent import gradient_ascent
from stochastic_gradient_ascent import stochastic_gradient_ascent
from newton_method import newton_method

def main():
    max_iter_count          = 1000000
    step_alpha              = 0.0001
    data_array, label_array = load_data("logistic_regression_data-1.txt")
    # theta, cost_v           = gradient_ascent(data_array, label_array, step_alpha, max_iter_count)
    theta, cost_v           = stochastic_gradient_ascent(data_array, label_array, step_alpha, max_iter_count)
    # theta, cost_v           = newton_method(data_array, label_array, max_iter_count=max_iter_count)
    print(theta)
    data_numbers            = np.shape(data_array)[0]

    data_x1_0 = []
    data_x1_1 = []
    data_x2_0 = []
    data_x2_1 = []

    for i in range(data_numbers):
        if label_array[i][0] == 0:
            data_x1_0.append(data_array[i][0])
            data_x2_0.append(data_array[i][1])
        else:
            data_x1_1.append(data_array[i][0])
            data_x2_1.append(data_array[i][1])

    figure = plt.figure("Result")

    fig_plot = figure.add_subplot(111)
    fig_plot.scatter(data_x1_0, data_x2_0, s=30, c='red', marker='s') # plot 0
    fig_plot.scatter(data_x1_1, data_x2_1, s=30, c='green')

    x = np.arange(0, 100, 1)
    y = (-theta[2,0] - theta[0,0] * x) / theta[1,0]

    fig_plot.plot(x,y)

    plt.xlabel('x1')
    plt.ylabel('x2')

    cost_fig  = plt.figure("Cost")
    cost_plot = cost_fig.add_subplot(111)
    epoches   = np.arange(0, max_iter_count+1, 1)
    cost_plot.plot(epoches, cost_v)

    plt.show()

if __name__=="__main__":
    main()

