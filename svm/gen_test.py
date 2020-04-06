import os
import shutil
from PIL import Image

def gen_test():
    input_path  = "./picture"
    output_path = "picture_256_256_test"
    label_file  = os.path.join(output_path,"label.txt")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    label_f     = open(label_file, 'a')

    for picture in os.listdir(input_path):
        picture_filepath = os.path.join(input_path, picture)
        name_target      = os.path.join(output_path, os.path.splitext(os.path.basename(picture))[0]+"-test.jpg")
        image            = Image.open(picture_filepath)
        image_new        = image.resize((256, 256), Image.ANTIALIAS).convert('L').rotate(180) 
        image_new.save(name_target)
        label            = (float(int(os.path.splitext(os.path.basename(picture))[0]) > 12) - 0.5 ) * 2 # Add Train Label
        label_f.write(os.path.basename(name_target)+'\t'+str(label)+'\n')

    
    print("#info: Done Gen Test!")

if __name__ == "__main__":
    gen_test()