#! /usr/bin/python3

import numpy as np
from sklearn import svm
from load_data import load_data

def train(data_mat, label_mat):
    clf = svm.SVC()
    clf.fit(data_mat, label_mat)
    return clf

if __name__ == "__main__":
    data_mat, label_mat = load_data("picture_256_256", "picture_256_256/label.txt")
    clf                 = train(data_mat, label_mat)
    test_data           = data_mat
    refer_label         = label_mat
    test_result         = clf.predict(test_data)
    print(np.mat(test_result))
    print(np.mat(refer_label).T)
