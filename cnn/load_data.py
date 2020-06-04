#! /usr/bin/python3

import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def read_image(file_name):
    im   = Image.open(file_name).convert('L')
    data = np.array(im)
    return data

def load_data(file_path):
    images = []
    labels = []
    for image in os.listdir(file_path):
        if image.endswith('.png'):
            image_data  = read_image(os.path.join(file_path,image))
            images.append(image_data)
            label = int(os.path.splitext(image)[0].split('_')[0])

            labels.append(label)
    
    data   = np.array(images)
    labels = np.array(labels)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=30)
    
    return data_train, data_test, labels_train, labels_test
