#! /usr/bin/python3

import os
import shutil
from PIL import Image

def im_resize():
    input_path  = "dataset"
    output_path = "dataset_resize"
    label_file  = os.path.join(output_path,"label.txt")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    labels_f = open(label_file, 'a')

    count_flag = [0, 0, 0]
    for picture in os.listdir(input_path):
        image     = Image.open(os.path.join(input_path, picture)).convert('L')
        resize_im = image.resize((256, 256))

        if os.path.splitext(picture)[-1] == ".bmp":
            label = 0
        elif int(os.path.splitext(picture)[0]) < 1200:
            label = 1
        else:
            label = 2
        
        resize_im.save(os.path.join(output_path, str(label) + "_" + str(count_flag[label]).zfill(3) + ".png"))
        labels_f.write(str(label) + "_" + str(count_flag[label]).zfill(3) + ".png" + '\t' + str(label) + '\n')

        count_flag[label] += 1

    print("Info: Resize Images Done!")

if __name__ == "__main__":
    im_resize()