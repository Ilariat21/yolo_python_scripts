#! /usr/bin/env python

import os
import argparse
import json
import cv2
import copy
from utils.utils import get_yolo_boxes, makedirs
from keras.models import load_model

def get_cords(image, boxes, labels, obj_thresh, quiet=True):
    x_center = []
    y_center = []
    width = []
    height = []
    label = []
    img_height, img_width, _ = image.shape

    for box in boxes:
        label_str = ''
        label_ = -1
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_ = i
            if not quiet: print(label_str)
        if label_ >= 0:
            x_center.append(round(((box.xmax+box.xmin)/2/img_width), 6))
            y_center.append(round(((box.ymax+box.ymin)/2/img_height), 6))
            width.append(round((box.xmax-box.xmin)/img_width, 6))
            height.append(round((box.ymax-box.ymin)/img_height, 6))
            label.append(str(label_))

    return x_center, y_center, width, height, label

def _main_(args):
    config_path  = args.conf
    input_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.8, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])
    
    ###############################
    #   Predict bounding boxes 
    ###############################
    image_paths = []
    filename = []

    if os.path.isdir(input_path):
        path = os.listdir(input_path)
        for i in range(len(path)):
            if os.path.isdir("{}{}".format(input_path, path[i])):  
                for inp_file in os.listdir("{}{}".format(input_path, path[i])):
                    image_paths.append("{}{}/{}".format(input_path, path[i], inp_file))
                    filename += [inp_file] 
            else:
                image_paths.append("{}{}".format(input_path, path[i]))
                filename += [path[i]]
    else:
        image_paths += [input_path]
        filename += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    filename = [inp_file for inp_file in filename if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    filename_new = copy.deepcopy(filename)

    for i in range(len(filename)):
        print(filename[i])
        filename[i] = str(filename[i]).replace("jpg", "txt") 
        filename[i] = str(filename[i]).replace("png", "txt") 
        filename[i] = str(filename[i]).replace("JPEG", "txt")

        # the main loop
    for i in range(len(image_paths)):
        image = cv2.imread(image_paths[i])
        print(image_paths[i])

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
        
        # get annotations on the image using bounding box coordinates 
        x_center, y_center, width, height, label = get_cords(image, boxes, config['model']['labels'], obj_thresh)

        # create txt file
        image_paths[i] = image_paths[i].replace(str(filename_new[i]), "")
        my_file = open("{}{}".format(image_paths[i], filename[i]), "w")  
        print("{}{}".format(image_paths[i], filename[i]))

        # write annotations to file
        for i in range(len(x_center)): 
            my_file.write("{} {} {} {} {} \n".format(label[i], x_center[i], y_center[i], width[i], height[i]))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')      
    
    args = argparser.parse_args()
    _main_(args)
