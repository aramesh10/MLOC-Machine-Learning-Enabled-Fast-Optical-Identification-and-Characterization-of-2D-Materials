import os
import numpy as np
import matplotlib.image as img
import albumentations as albu

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

NUM_OF_CLASSES = 3

class GrapheneDataset(Dataset):
    
    def __init__(self, img_dir, label_dir, preprocess_bool=False):
        self.img = np.array([rgb2gray(img.imread(img_dir+image)) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])  

        if label_dir != None:
            self.label = np.array([img.imread(label_dir+label) for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])
            assert(len(self.img) == len(self.label))
        else:
            self.label = None    
        
        self.preprocess_bool = preprocess_bool
        if preprocess_bool:
            self.transform = self.preprocessing()
        else:
            self.transform = None

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.label[idx]


        # Image transformations
        if self.preprocess_bool:
            # Image transformations
            image = self.transform(image=image)["image"]
            image = self.image_normalize(image=image)
        image = self.image_to_tensor(image=image)

        # Label transformations
        label = self.label_to_tensor(label=label)
        
        return (image, label)

    def preprocessing(self):
        transform = [
            albu.Sharpen(alpha=(0,1), p=1)
        ]
        return albu.Compose(transform)

    def image_to_tensor(self, image):
        return torch.tensor(image).float().reshape(1, image.shape[0], image.shape[1])

    def image_normalize(self, image):
        img_norm = normalize(image)
        return img_norm

    def label_to_tensor(self, label):
        return torch.tensor(255*label-1).reshape(1, label.shape[0], label.shape[1]).to(torch.int64)

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])