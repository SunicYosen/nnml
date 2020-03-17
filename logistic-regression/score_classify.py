# 
# Score classify
# 

import numpy as np 
import matplotlib.pyplot as plt

from load_data import load_data
from functions import sigmoid
from gradient_ascent import gradient_ascent
from stochastic_gradient_ascent import stochastic_gradient_ascent
from newton import newton_method

def main():
    data_array, label_array = load_data("logistic_regression_data-1.txt")
    # weights               = gradient_ascent(data_array, label_array)
    # weights              = stochastic_gradient_ascent(data_array, label_array)
    weights               = newton_method(data_array, label_array)
    print(weights)
    data_numbers          = np.shape(data_array)[0]

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

    figure = plt.figure()

    fig_plot = figure.add_subplot(111)
    fig_plot.scatter(data_x1_0, data_x2_0, s=30, c='red', marker='s') # plot 0
    fig_plot.scatter(data_x1_1, data_x2_1, s=30, c='green')

    x = np.arange(0, 100, 1)
    y = (-weights[2,0] - weights[0,0] * x) / weights[1,0]

    fig_plot.plot(x,y)

    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

if __name__=="__main__":
    main()

