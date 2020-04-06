#! /usr/bin/python3

import os
import shutil
from PIL import Image


def image_gray(file_name):
    image = Image.open(file_name).convert('L')
    return image

# input_path: origion picture file name,
# output_path: output picture directory,
# width: target width,
# height: target height
def image_cut(input_name, output_path, width, height, label_file):
    file_name = input_name
    im        = Image.open(file_name)
    (origion_w, origion_h) = im.size  # Get origion size of picture
    target_w = width
    target_h = height

    label    = (float(int(os.path.splitext(os.path.basename(file_name))[0]) > 12) - 0.5 ) * 2 # Add Train Label
    label_f  = open(label_file, 'a')

    for w in range(int(origion_w / target_w)):
        for h in range (int(origion_h / target_h)):
            name_target = os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0]+'-'+str(w)+'-'+str(h)+".jpg")
            print(name_target)
            cm = im.crop(box=(w*target_w, h*target_h, (w+1)*target_w, (h+1)*target_h)).convert('L')
            cm.save(name_target)
            label_f.write(os.path.basename(name_target)+'\t'+str(label)+'\n')

            if (origion_w > (w+1) * target_w) & ((w+1) == int(origion_w/target_w)) & (origion_h > (h+1) * target_h) & ((h+1) == int(origion_h/target_h)):
                name_target = os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0]+'-'+str(w+1)+'-'+str(h+1)+".jpg")
                print(name_target)
                cm = im.crop(box=((origion_w-target_w), (origion_h-target_h), origion_w, origion_h)).convert('L')
                cm.save(name_target)
                label_f.write(os.path.basename(name_target)+'\t'+str(label)+'\n')

            if (origion_w > (w+1) * target_w) & ((w+1) == int(origion_w/target_w)):
                name_target = os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0]+'-'+str(w+1)+'-'+str(h)+".jpg")
                print(name_target)
                cm = im.crop(box=((origion_w-target_w), h*target_h, origion_w, (h+1)*target_h)).convert('L')
                cm.save(name_target)
                label_f.write(os.path.basename(name_target)+'\t'+str(label)+'\n')

            if (origion_h > (h+1) * target_h) & ((h+1) == int(origion_h/target_h)):
                name_target = os.path.join(output_path, os.path.splitext(os.path.basename(file_name))[0]+'-'+ str(w)+'-'+str(h+1)+".jpg")
                print(name_target)
                cm = im.crop(box=((w)*target_w, (origion_h-target_h), (w+1)*target_w, origion_h)).convert('L')
                cm.save(name_target)
                label_f.write(os.path.basename(name_target)+'\t'+str(label)+'\n')

    label_f.close()

def run_cut():
    input_path = "./picture"
    output_path = "picture_256_256"
    label_file  = os.path.join(output_path,"label.txt")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for picture in os.listdir(input_path):
        picture_filepath = os.path.join(input_path, picture)
        image_cut(picture_filepath, output_path, width=256, height=256, label_file=label_file)
    
    print("#info: Done Cut!")

if __name__ == "__main__":
    run_cut()
            
