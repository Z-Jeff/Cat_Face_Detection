import os
import sys
#sys.path.append(os.getcwd())
#from wider_loader import WIDER
import cv2
import time
import numpy as np
import pandas as pd

"""
 modify .mat to .txt 
"""

# original images path
#path_to_image = './data_set/face_detection/WIDERFACE/WIDER_train/WIDER_train/images'
path_to_image = '../../data/data18748/train'

# csv label file path
#file_to_label = './data_set/face_detection/WIDERFACE/wider_face_split/wider_face_split/wider_face_train.mat'
file_to_label = '../../data/data18748/train.csv'

#target file path
target_file = './anno_store/anno_train.txt'

#wider = WIDER(file_to_label, path_to_image)


def get_points(path, csv_data):
    image_indices = int(path.split('/')[-1].split('.')[0])
    data = csv_data.values[image_indices]
    left_eye = [data[1], data[2]]
    right_eye = [data[3], data[4]]
    mouth = [data[5], data[6]]
    left_ear1 = [data[7], data[8]]
    left_ear2 = [data[9], data[10]]
    left_ear3 = [data[11], data[12]]
    right_ear1 = [data[13], data[14]]
    right_ear2 = [data[15], data[16]]
    right_ear3 = [data[17], data[18]]
    points = [left_eye, right_eye, mouth, left_ear1, left_ear2, left_ear3, right_ear1, right_ear2, right_ear3]
    return points

def generate_box(points, im_width, im_height):
    delt_ear = np.abs(points[7][0] - points[4][0])
    delt_ear_mouth = np.abs((points[7][1]+points[4][1])/2 - points[2][1])
    
    x_min = points[0][0]
    x_max = x_min
    y_min = points[0][1]
    y_max = y_min
    for i in range(1, 9):
        if(points[i][0] < x_min):
            x_min = points[i][0]
        if(points[i][0] > x_max):
            x_max = points[i][0]
        if(points[i][1] < y_min):
            y_min = points[i][1]
        if(points[i][1] > y_max):
            y_max = points[i][1] 
    aera = np.abs(x_max-x_min) * np.max(y_max-y_min)
    aera_scale = aera / (im_width*im_height)
    scale = -aera_scale + 1
    
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    sinTheta = (y2-y1) / np.sqrt(np.square(x2-x1) + np.square(y2-y1))
    #print(sinTheta)
    
    x_min -= delt_ear * 0.2*sinTheta
    x_max -= delt_ear * 0.2*sinTheta
    
    x_min -= delt_ear * 0.2*scale
    x_max += delt_ear * 0.2*scale
    y_min -= delt_ear_mouth * 0.1*scale
    y_max += delt_ear_mouth * 0.3*scale
    
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > im_width:
        x_max = im_width
    if y_max > im_height:
        y_max = im_height
        
    return [x_min, x_max, y_min, y_max]

if __name__ == '__main__':
    line_count = 0
    print('start transforming....')
    t = time.time()
    
    csv_data = pd.read_csv(file_to_label)
    
    image_paths = os.listdir(path_to_image)
    image_paths = [os.path.join(path_to_image, path) for path in image_paths]
    image_num = len(image_paths)
    
    with open(target_file, 'w+') as f:
        # press ctrl-C to stop the process
        
        for i in range(image_num):
            points = get_points(image_paths[i], csv_data)
            image = cv2.imread(image_paths[i])
            image_h, image_w = image.shape[0], image.shape[1]
            
            [x_min, x_max, y_min, y_max] = generate_box(points, image_w, image_h)
            
            line = '%s %d %d %d %d ' % (image_paths[i], x_min, y_min, x_max, y_max)
            for p in points:
                line += '%s %s ' % (p[0], p[1])
            line += '\n'
            
            line_count += 1
            
            if i % 1000 == 0:
                print('transforming %d image label' % i)
            f.write(line)

    st = time.time()-t
    print('end transforming')
    
    print('spend time: %ds'%st)
    print('total line(images):%d'%line_count)
    #print('total boxes(faces):%d'%box_count)
