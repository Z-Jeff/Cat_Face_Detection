import os
import cv2
import sys
import random 
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras import regularizers
from tensorflow.keras import Model

print('tf.version:', tf.__version__)

TRAIN_VAL_SPLIT = 1
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_dir = './dataset/train/'
test_dir = './dataset/test/'
csv_file = './dataset/train.csv'


# Get file paths
train_paths = pathlib.Path(train_dir)
train_paths = list(train_paths.glob('*'))
train_paths = [str(path) for path in train_paths]
val_paths = []
if TRAIN_VAL_SPLIT:
    val_paths = train_paths[-3000:]
    train_paths = train_paths[:-3000]
train_num = len(train_paths)
val_num = len(val_paths)
print('train num:     ', train_num)
print('validation num:', val_num)
print(train_paths[:3])

# Get laabls
csv_data = pd.read_csv(csv_file)
def get_labels(image_path, csv_data):
    image_indices = int(image_path.split('/')[-1].split('.')[0])
    points_data = csv_data.values[image_indices]
    points_data = points_data[1:]
    return points_data
train_labels = [get_labels(path, csv_data) for path in train_paths]    
val_labels = [get_labels(path, csv_data) for path in val_paths]    


# Create a dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))   
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))   
def load_train_data(path, labels):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])     
    #image /= 255.0
    return image, labels

def load_val_data(path, labels):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])     
    image /= 255.0
    return image, labels
    
train_ds = train_ds.map(load_train_data)
val_ds = val_ds.map(load_val_data)

'''
for image, label in train_ds.take(1):
    print(image)
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
'''
img, l = load_train_data('./dataset/train/6797.jpg', 1)
print(img)

im = cv2.imread('./dataset/train/6797.jpg')
print(im)
'''
def prepare_for_training(ds, cache=True, shuffle_buffer_size=500):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
          ds = ds.cache(cache)
        else:
          ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds   

dataset = prepare_for_training(ds)
'''
