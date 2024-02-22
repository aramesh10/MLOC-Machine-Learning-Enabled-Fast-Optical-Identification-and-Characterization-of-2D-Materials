import os
import matplotlib.image as img

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

NUM_OF_CLASSES = 3
SIZE = 32*3

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class GrapheneDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, label_transform=None):
        self.img = np.array([rgb2gray(img.imread(img_dir+image)) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])  
        self.img = self.img[:, :SIZE, :SIZE].reshape(self.img.shape[0], 1, SIZE, SIZE)

        if label_dir != None:
            self.label = np.array([img.imread(label_dir+label) for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])
            self.label = self.label[:, :SIZE, :SIZE]
            assert(len(self.img) == len(self.label))
        else:
            self.label = None    
        
        self.transform = transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        if self.transform != None:
            img = self.transform(img)
        img = torch.tensor(img).float()
        
        label = self.label[idx]
        if self.label_transform != None:
            label = self.label_transform(label)
        label = torch.tensor(255*label-1).reshape(1, SIZE, SIZE).to(torch.int64)
        
        return (img, label)
    
def post_process_mask_prediction(prediction_mask):
    post_process_mask = reorder_arr(np.squeeze(prediction_mask), (96, 96, 3))
    post_process_mask = reverse_one_hot(post_process_mask)
    return post_process_mask

    
def reorder_arr(arr, shape):
    assert(len(shape) == 3)
    reorder_arr = []
    for i in range(shape[0]):
        temp1 = []
        for j in range(shape[1]):
            append_arr = [arr[0][i][j], arr[1][i][j], arr[2][i][j]]
            temp1.append(append_arr)
        reorder_arr.append(temp1)
    reorder_arr = np.array(reorder_arr)
    return reorder_arr

def reverse_one_hot(arr):
    arr_shape = arr.shape
    reverse_one_hot_arr = np.empty((arr_shape[0], arr_shape[1]))
    for row in range(arr_shape[0]):
        for col in range(arr_shape[1]):
            reverse_one_hot_arr[row][col] = np.argmax(arr[row][col])
    return reverse_one_hot_arr
