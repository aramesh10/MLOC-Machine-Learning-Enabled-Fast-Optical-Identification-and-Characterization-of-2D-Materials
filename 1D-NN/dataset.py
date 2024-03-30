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
        # Get data
        x_train = np.array([rgb2gray(img.imread(img_dir+image)) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])     # Shape: [N, W, H]
        y_train = np.array([img.imread(label_dir+label)*255 for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])     # Shape: [N, W, H]

        # Segment data into rows
        self.rows = np.reshape(x_train, (x_train.shape[0]*x_train.shape[2], x_train.shape[1]))        # Shape: [N*H, 1, W]
        self.labels = np.reshape(y_train, (y_train.shape[0]*y_train.shape[2], y_train.shape[1]))      # Shape: [N*H, 1, W]

        self.preprocessing_bool = preprocess_bool

    def __len__(self):
        # print(self.rows.shape)
        return self.rows.shape[0]

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = self.labels[idx]

        # Image transformations
        if self.preprocessing_bool:
            row = self.row_normalize(row=row.reshape(-1, 1))
        row = self.row_to_tensor(row=row)

        # Label transformations
        label = self.label_to_tensor(label=label)
        # label = one_hot(label, num_classes=NUM_OF_CLASSES)
        # label = label.reshape(label.shape[1],label.shape[0])
        
        return (row, label)

    def preprocessing(self):
        transform = [
            albu.Sharpen(alpha=(0,1), p=1)
        ]
        return albu.Compose(transform)

    def row_to_tensor(self, row):
        return torch.tensor(row).float().reshape(1, row.shape[0])

    def row_normalize(self, row):
        return normalize(row, axis=0)

    def label_to_tensor(self, label):
        return torch.tensor(label-1).reshape(label.shape[0]).to(torch.int64)

# def post_process_mask_prediction(prediction_mask):
#     post_process_mask = reorder_arr(np.squeeze(prediction_mask), (96, 96, 3))
#     post_process_mask = reverse_one_hot(post_process_mask)
#     return post_process_mask
    
# def reorder_arr(arr, shape):
#     assert(len(shape) == 3)
#     reorder_arr = []
#     for i in range(shape[0]):
#         temp1 = []
#         for j in range(shape[1]):
#             append_arr = [arr[0][i][j], arr[1][i][j], arr[2][i][j]]
#             temp1.append(append_arr)
#         reorder_arr.append(temp1)
#     reorder_arr = np.array(reorder_arr)
#     return reorder_arr

# def reverse_one_hot(arr):
#     arr_shape = arr.shape
#     reverse_one_hot_arr = np.empty((arr_shape[0], arr_shape[1]))
#     for row in range(arr_shape[0]):
#         for col in range(arr_shape[1]):
#             reverse_one_hot_arr[row][col] = np.argmax(arr[row][col])
#     return reverse_one_hot_arr

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])