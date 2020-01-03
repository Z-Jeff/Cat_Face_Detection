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

MODEl_INDEX = 2

TRAIN_VAL_SPLIT = 1
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 1

train_dir = '../../data/data18748/train/'
test_dir = '../../data/data18748/test/'
csv_file = '../../data/data18748/normalize_train.csv'


###----- Get file paths -----###
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
#print(train_paths[:3])


###----- Get labels -----###
csv_data = pd.read_csv(csv_file)
def get_labels(image_path, csv_data):
    image_indices = int(image_path.split('/')[-1].split('.')[0])
    points_data = csv_data.values[image_indices]
    points_data = points_data[1:]
    return points_data
train_labels = [get_labels(path, csv_data) for path in train_paths]    
val_labels = [get_labels(path, csv_data) for path in val_paths]    


###----- Create datasets -----###
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))   
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))   
def load_train_data(path, labels):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=3)
    
    #[h, w] = [image.numpy().shape[0], image.numpy().shape[1]]
    print('shape', tf.shape(image))
    #for i in range(9):
    #    labels[i*2] = labels[i*2]/w
    #    labels[i*2+1] = labels[i*2+1]/h
    
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])     
    image /= 255.0
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
AUTOTUNE = tf.data.experimental.AUTOTUNE
def prepare_for_training(ds, cache=True, shuffle=True, shuffle_buffer_size=500):
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

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds, shuffle=False)


###----- Create the model -----###
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(18)
])
print(model.summary())
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse', 'mae'])
              
###----- Train the model -----###
history = model.fit_generator(
    train_ds,
    steps_per_epoch=train_num // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=val_num // BATCH_SIZE
)
model_save_dir = './model/' + str(MODEl_INDEX) + '/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model.save(model_save_dir, save_format = 'tf')