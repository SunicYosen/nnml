#! /usr/bin/python3

import numpy as np
from sklearn import svm
from load_data import load_data

def train(data_mat, label_mat):
    clf = svm.SVC(C=float(0.01),kernel='linear',verbose=True, tol=1)
    clf.fit(data_mat, label_mat)
    return clf

if __name__ == "__main__":
    data_mat, label_mat = load_data("picture_256_256", "picture_256_256/label.txt")
    test_data_mat, test_label_ref = load_data("picture_256_256_test", "picture_256_256_test/label.txt")
    clf                 = train(data_mat, np.array(label_mat.ravel().T))
    test_result         = clf.predict(test_data_mat)
    delta               = test_result - test_label_ref.T
    errors              = np.nonzero(delta)
    print(clf)
    print(delta)
