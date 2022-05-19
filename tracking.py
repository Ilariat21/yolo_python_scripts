#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.colors import get_color
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

def mod(rows, cols, a, b, count):
    unadded_cols = []
    unadded_rows = []
    for i in range(len(count)):
        count[i]+=1
    if len(a)>len(b):
        c=copy.deepcopy(a)
        for i in range(len(rows)):
            c[rows[i]][1]=b[cols[i]][1]
            count[rows[i]]=0
        for i in range(len(b)):
            if i not in rows and count[b[i][0]]<10:
                c[i][1]=a[c[i][0]][1]
            elif i not in rows and count[b[i][0]]>=10:
                c.pop(i)
                count[b[i][0]]=0
    else:
        c=copy.deepcopy(b)
        for i in range(len(rows)):
            c[rows[i]][1]=b[cols[i]][1]
        for i in range(len(b)):
            count[b[i][0]]=0
            if i not in cols:
                unadded_cols.append(i)
            if i not in rows:
                unadded_rows.append(i)
        for i in range(len(unadded_rows)):
            c[unadded_rows[i]][1]=b[unadded_cols[i]][1]
    return c, count

def modify_list(a, b, count):
    new_a, new_b = np.zeros((len(a), 2)), np.zeros((len(b), 2))
    for i in range(len(a)):
        new_a[i] = a[i][1]
    for i in range(len(b)):
        new_b[i]=b[i][1]
    cost = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            cost[i][j]=np.linalg.norm(new_a[i]-new_b[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    #print(row_ind, col_ind)
    b, count = mod(row_ind, col_ind, a, b, count)
    return b, count

def draw_lines(allList, image, labels):
    for i in range(len(labels)):
        if labels[i]=="person":
            line_color = get_color(i)
    for m in range (1, len(allList)):
        for i in range (len(allList[m-1])):
            for j in range(len(allList[m])):
                if allList[m-1][i][0]==allList[m][j][0]:
                    cv2.line(img=image, 
                             pt1 = (int(allList[m-1][i][1][0]), int(allList[m-1][i][1][1])), 
                             pt2 = (int(allList[m][j][1][0]), int(allList[m][j][1][1])), 
                             color = line_color, thickness = 5)

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    cords = []
    count = 0
    for box in boxes:
        label_str = ''
        label = -1
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if labels[i]=="person":
                    if label_str != '': label_str += ', '
                    label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                    label = i
                else:
                    label = -1
            if not quiet: print(label_str)
                
        if label >= 0:
            cords.append([count, [(box.xmin+box.xmax)/2, box.ymax]])
            count += 1
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        
    return image, cords

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output
    max_people_perframe = 8
    superlist_size = 15

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    
    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)
        count = []
        for i in range(max_people_perframe):
            count.append(0)
        superlist = []
        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    image, cords = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)  

                    if len(superlist)!=0:
                        cords, count = modify_list(superlist[len(superlist)-1], cords, count)
                        if len(superlist)==superlist_size:
                            superlist.pop(0)
                    superlist.append(cords)
                    
                    draw_lines(superlist, image, config['model']['labels']) 

                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4': # do detection on a video  
        count = []
        for i in range(max_people_perframe):
            count.append(0)
        superlist = []
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        image, cords = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)  

                        if len(superlist)!=0:
                            cords, count = modify_list(superlist[len(superlist)-1], cords, count)
                            if len(superlist)==superlist_size:
                                superlist.pop(0)
                        superlist.append(cords)
                        
                        draw_lines(superlist, image, config['model']['labels'])

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()
        print(superlist)
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
