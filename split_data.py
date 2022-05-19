import os
import shutil
import random

path1 = r'/Users/Tairali/Desktop/all_data/img' #path to img folder
path2 = r'/Users/Tairali/Desktop/all_data/annot' #path to annot folder

output  = r'/Users/Tairali/Desktop/all_data_Tair'

num_data = len(os.listdir(path2))
valid_train_ratio = 20 #the percent of all data that will be valid

def check(dir):
    if not os.path.exists(dir):
        os.makedirs(dir) 

def makedir(output):
    check("{}/train_img".format(output))
    check("{}/valid_img".format(output))
    check("{}/train_annot".format(output))
    check("{}/valid_annot".format(output))
    a = "{}/train_img".format(output)
    b = "{}/valid_img".format(output)
    c = "{}/train_annot".format(output)
    d = "{}/valid_annot".format(output)
    return a, b, c, d

valid = random.sample(range(num_data), int(num_data/(100/valid_train_ratio)))

images = os.listdir(path1)
annotations = os.listdir(path2)

check(output)

target1, target2, target3, target4 = makedir(output)
print("All necessary folders are created")
print("In progress...")
for i in range(num_data):
    if i in valid:
        shutil.copyfile(('{0}/{1}'.format(path1, images[i])), '{0}/{1}'.format(target2, images[i]))
        shutil.copyfile(('{0}/{1}'.format(path2, annotations[i])), '{0}/{1}'.format(target4, annotations[i]))
    else:
        shutil.copyfile(('{0}/{1}'.format(path1, images[i])), '{0}/{1}'.format(target1, images[i]))
        shutil.copyfile(('{0}/{1}'.format(path2, annotations[i])), '{0}/{1}'.format(target3, annotations[i]))
print("Done")