#! /usr/bin/python3
from load_data import load_data
from train import train
from test import test

def main():
    C = float('0.015')
    # C = float('inf')
    toler = 0.00005
    max_iter = 100
    kernel_type = ('linear', 0)
    # kernel_type = ('rbf', 50)

    data_mat, label_mat = load_data("picture_256_256", "picture_256_256/label.txt")
    support_vector_data, support_vector_label, support_vector_alphas, b = \
        train(data_mat, label_mat, C=C, toler=toler, max_iter=max_iter, kernel_type=kernel_type)
    print(support_vector_alphas)
    print(support_vector_label)
    test_data_mat, test_label_ref = load_data("picture_256_256_test", "picture_256_256_test/label.txt")
    total_count, errot_count      = test(test_data_mat, test_label_ref, support_vector_data, support_vector_label, support_vector_alphas, b, kernel_type)
    return 0

if __name__ == "__main__":
    main()
