import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

train_dir = 'work/Cat_Face_Detection/dataset/train/'
test_dir = 'work/Cat_Face_Detection/dataset/test/'
csv_file = 'work/Cat_Face_Detection/dataset/train.csv'

image_paths = os.listdir(train_dir)
image_paths = [train_dir + path for path in image_paths]
image_num = len(image_paths)
print(image_num)

csv_data = pd.read_csv(csv_file)

def get_points(path):
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
     
def generate_box(points):
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
        x_min -= delt_ear * 0.2
        x_max += delt_ear * 0.2
        y_min -= delt_ear_mouth * 0.1
        y_max += delt_ear_mouth * 0.3

        return [x_min, x_max, y_min, y_max]
        
def show_points(path):
    points = get_points(path)
    image = mpimg.imread(path)
    fig, ax = plt.subplots(1)
    plt.imshow(image)
    plt.plot(points[0][0],points[0][1], 'co')
    plt.plot(points[1][0],points[1][1], 'co')
    plt.plot(points[2][0],points[2][1], 'co')
    plt.plot(points[3][0],points[3][1], 'co')
    plt.plot(points[4][0],points[4][1], 'co')
    plt.plot(points[5][0],points[5][1], 'co')
    plt.plot(points[6][0],points[6][1], 'co')
    plt.plot(points[7][0],points[7][1], 'co')
    plt.plot(points[8][0],points[8][1], 'co')
    
    [x_min, x_max, y_min, y_max] = generate_box(points)
       
    '''
    if(x_min < 0):
        x_min = 0
    if(x_max > image.shape[0]):
        x_max = image.shape[0]
    if(y_min < 0):
        y_min = 0
    if(y_max > image.shape[1]):
        y_max = image.shape[1]
    '''
    
    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    

show_points(image_paths[944])    
    