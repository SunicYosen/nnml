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
        line_array = line.strip().split(' ')
        temp_array = []
        for index in range(len(line_array) - 1):
            temp_array.append(float(line_array[index]))
        data_array.append(temp_array)
        # Read area & bedrooms

        label_array.append([int(line_array[len(line_array) - 1])])
        # Read Price

    return data_array, label_array

# 
# Test load data function
#
def test_load_data():

    data_array, label_array = load_data("data_train.txt")

    print(data_array)
    print(label_array)

if __name__ == "__main__":
    test_load_data()
