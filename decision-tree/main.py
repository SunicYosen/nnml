#! /usr/bin/python3
import numpy as np
from load_data import load_data
from decision_tree import create_tree
from plot_tree import create_plot

def main():
    features, data_set = load_data('data.txt')
    features_d         = features
    data_set_d         = data_set

    del(features_d[0])           # Remove Symbol
    for example in data_set_d:
        del(example[0])
    
    decision_t         = create_tree(data_set_d, features_d)
    print(decision_t)
    create_plot(decision_t)

if __name__=='__main__':
    main()