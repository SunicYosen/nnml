#! /usr/bin/python3
# -*- UTF-8 -*-

def load_data(file_name):

    features_name   = []
    features_array  = []

    file_point = open(file_name)
    features_name = file_point.readline().strip().split(' ')

    for line in file_point.readlines():
        line_array = line.strip().split(' ')
        features_array.append(line_array)

    return features_name, features_array
    print("INFO: Load data DONE!\n")
    

if __name__=='__main__':
    name, data = load_data("data.txt")
    print(name, data)
    