import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms as T

# Reference:
# https://github.com/chenyuntc/pytorch-best-practice/blob/master/main.py
# https://ptorch.com/news/215.html

BATCH_SIZE = 5
IMG_SIZE = 224
TRAIN_VAL_SPLIT = True

train_dir = '../../data/data18748/train/'
test_dir = '../../data/data18748/test/'
label_csv = '../../data/data18748/train.csv'

###--- Create a dataste ---###
class myDataset(Dataset):
    def __init__(self, root, label_csv, train_all=True, transforms=None):
        self.val = val
        self.augment = augment
        self.csv_data = pd.read_csv(csv_file)
        
        image_paths = np.array([x.path for x in os.scandir(root)])
        image_num = len(image_paths)
        
        if train_all:
            self.image_paths = image_paths
        else:
            self.image_paths = image_paths[:int(0.7*image_num)]
            
        # shuffle
        np.random.seed(100)
        self.image_paths = np.random.permutation(self.image_paths)
        
        if transforms is None:
            # Using the mean and std of Imagenet
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                    T.Resize(IMG_SIZE),
                    #T.CenterCrop(224),
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])
                    
    def __getitem__(self, index):
        # Return one image per time
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        img = self.transforms(img)
        
        labels = csv_data.values[image_indices][1:]
        return img, labels
        
    def __len__(self):
        return len(self.image_paths)

'''
train_dataset = myDataset(train_dir, label_csv)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers：使用多进程加载的进程数，0代表不使用多进程

for image in train_loader:
    image = image.to(device)
'''
    
'''
###--- Defeine the network ---###
class Net(nn.Module):
    def __init__(self, label_num=18):
        super(SimpleNet, self).__init__()
 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
 
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
 
        self.pool = nn.MaxPool2d(kernel_size=2)
 
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
 
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
 
        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=label_num)
 
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
 
        output = self.conv2(output)
        output = self.relu2(output)
 
        output = self.pool(output)
 
        output = self.conv3(output)
        output = self.relu3(output)
 
        output = self.conv4(output)
        output = self.relu4(output)
 
        output = output.view(-1, 1) # like Flatten() in keras
 
        output = self.fc(output)
 
        return output
'''

