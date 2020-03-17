#! /usr/bin/python3

#
# Load data function
# file_name: the data file name
# 
def load_data(file_name):

    data_array  = []
    label_array = []

    file_point = open(file_name)

    for line in file_point.readlines():
        line_array = line.strip().split(',')
        data_array.append([float(line_array[0]), float(line_array[1]), 1.0]) 
        # 1.0 constant parameter
        # shape: 100 * 3

        label_array.append([int(line_array[2])])
        # shape: 100 * 1

    return data_array, label_array

# 
# Test load data function
#
def test_load_data():

    data_array, label_array = load_data("logistic_regression_data-1.txt")

    print(data_array)
    print(label_array)

if __name__ == "__main__":
    test_load_data()
