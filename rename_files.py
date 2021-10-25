#! /usr/bin/env python

import os
import shutil

# renaming files (by iteration) and copying them into output folder

# Change the following 4 parameters: path to files to be renamed, output path, new file name (iteration will be added), file extension (ex: .jpg, .png. JPEG)

input_path = '/Users/user/Desktop/old_folder' # do not add last slash (/)
output_path = '/Users/user/Desktop/new_folder' # do not add last slash (/)
filename = "Name_of_my_files_"
extension = ".png" # do not forget to add dot before extension

# features: can find files in folder which is located in folder;
#           changing the extension of files;
#           taking only files of necessary extension (avoiding files of other extensions)

def check_existance(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_files(input_path):
    image_paths = []

    path = os.listdir(input_path)
    for i in range(len(path)):
        if os.path.isdir("{}/{}".format(input_path, path[i])):     
            for inp_file in os.listdir("{}/{}".format(input_path, path[i])):
                image_paths.append("{}/{}/{}".format(input_path, path[i], inp_file))
        else:
            image_paths.append("{}/{}".format(input_path, path[i]))

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-int(len(str(extension))):] in [extension])]

    return image_paths  

def start():
    check_existance(output_path)

    image_paths = get_files(input_path)

    print("In progress...")

    count = 0
    for paths in image_paths:
        shutil.copyfile(paths, '{0}/{1}0{2}{3}{4}'.format(output_path, filename, '0'*(len(str(len(image_paths)))-len(str(count))), str(count), extension))
        count += 1

    print("Done")

if __name__ == "__main__":
    start()
