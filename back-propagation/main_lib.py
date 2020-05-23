
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

from load_data import load_data

def main():
    # Load data
    data_file               = "data_train.txt"
    data_array, label_array = load_data(data_file)
    data_mat                = np.mat(data_array)
    label_mat               = np.mat(label_array).T
    # To 0-1
    data_mat  = preprocessing.MinMaxScaler().fit_transform(data_mat)
    label_mat = preprocessing.MinMaxScaler().fit_transform(label_mat)

    regreesor = MLPRegressor(hidden_layer_sizes=(10), activation='relu', solver='adam', alpha=0.001, tol=1e-8, max_iter=100000)

    regreesor.fit(data_mat, label_mat)
    predit_result = regreesor.predict(data_mat)

    print(label_mat)
    print(predit_result)

if __name__=='__main__':
    main()