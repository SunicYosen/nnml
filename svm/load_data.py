#! /usr/bin/python3
import os

from PIL import Image
import numpy as np

def load_image(file_name):
    image = Image.open(file_name)
    image_data = image.getdata()
    image_array = np.array(image_data, 'f')    # Reshape to matrix (height * width)
    return image_array

def load_data(image_path, label_file):
    label_f      = open(label_file)
    label_dict   = {}
    label_line   = label_f.readline()
    data_array   = []
    label_array  = []

    while label_line:
        label_array_temp = label_line.strip().split('\t')
        label_dict[label_array_temp[0]] = label_array_temp[-1]
        label_line = label_f.readline()

    for file in os.listdir(image_path):
        if (os.path.splitext(file)[1] == '.jpg') | (os.path.splitext(file)[1] == '.bmp'):
            image_name  = os.path.basename(file)
            image_array = load_image(os.path.join(image_path, file))
            image_label = [float(label_dict[image_name])]
            data_array.append(image_array)
            label_array.append(image_label)
    
    data_mat  = (np.mat(data_array) - 128 )/128.0
    label_mat = np.mat(label_array)
    # print(np.shape(data_mat),np.shape(label_mat))
    return data_mat, label_mat

# load_data("picture_256_256", "picture_256_256/label.txt")