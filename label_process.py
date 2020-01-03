from __future__ import division
import os
import cv2
import pandas as pd 

train_dir = '../../data/data18748/train/'
csv_file = '../../data/data18748/train.csv'
save_csv_file = '../../data/data18748/normalize_train.csv'

csv_data = pd.read_csv(csv_file)
train_num = len(os.listdir(train_dir))
print('train num:', train_num)

out_csv = {}
label = list(csv_data.columns.values)
for l in label:
    out_csv[l] = []

print(out_csv)

def get_labels(image_path, csv_data):
    image_indices = int(image_path.split('/')[-1].split('.')[0])
    points_data = csv_data.values[image_indices]
    points_data = points_data[1:]
    return points_data
    
    
for i in range(train_num):
    file_name = train_dir + str(i) + '.jpg'
    points_label = get_labels(file_name, csv_data)
    
    img = cv2.imread(file_name)
    [h, w] = [img.shape[0], img.shape[1]]
    
    out_csv[label[0]].append(i)
    for k in range(9):
        a = points_label[k*2]/float(w)
        b = points_label[k*2+1]/float(h)
        out_csv[label[k*2+1]].append(a)
        out_csv[label[k*2+2]].append(b)
        
    print(i, a, b)
    
    
df = pd.DataFrame(out_csv)
df.to_csv(save_csv_file, index=False)

    