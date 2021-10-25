import os
import random

#Valid train ratio = num of data in test set / num of all data
valid_train_ratio = 20

#Please write down all directories to the folder with images
path_to_img = '/Users/user/Desktop/darknet/custom_data/images'

list_img = os.listdir(path_to_img)

train_file = open("train.txt", "w") 
test_file = open("test.txt", "w") 

list_img = [inp_file for inp_file in list_img if (inp_file[-3:] in ["png"])]

num_data = len(list_img)

valid = random.sample(range(num_data), int(num_data/(100/valid_train_ratio)))

for i in range(num_data):
    if i in valid:
        test_file.write("{}/{}\n".format(path_to_img, list_img[i]))
    else:
        train_file.write("{}/{}\n".format(path_to_img, list_img[i]))