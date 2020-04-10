#! /usr/bin/python3

#
# Load data function
# file_name: the data file name
# 

def load_data(file_name):

    data_array  = []

    file_point = open(file_name)

    for line in file_point.readlines():
        line_array = line.strip().split(' ')
        data_array.append([float(line_array[0]), float(line_array[1])]) 

    return data_array

if __name__ == "__main__":
    data_array = load_data("data.txt")

    for i in data_array:
        print(i[1])

    print(data_array)