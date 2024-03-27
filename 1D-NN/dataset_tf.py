import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import albumentations as albu

import tensorflow as tf
import tensorflow_datasets as tfds

from torch.nn.functional import one_hot

SIZE = 128
NUM_OF_CLASSES = 3
class GrapheneDataset():
    def __init__(self, img_dir, label_dir, preprocess_bool=True):
        # Get data
        x_train = np.array([rgb2gray(img.imread(img_dir+image)) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])     # Shape: [N, W, H]
        y_train = np.array([img.imread(label_dir+label)*255 for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])     # Shape: [N, W, H]

        # Segment data into rows
        self.rows = np.reshape(x_train, (x_train.shape[0]*x_train.shape[2], x_train.shape[1]))        # Shape: [N*H, 1, W]
        self.labels = np.reshape(y_train, (y_train.shape[0]*y_train.shape[2], y_train.shape[1]))      # Shape: [N*H, 1, W]

        self.preprocessing_bool = preprocess_bool

    def __len__(self):
        return self.rows.shape[0]

    def __getitem__(self, idx):
        # Pad to correct size
        row = self.rows[idx]
        label = self.labels[idx]
        row = np.pad(row, (int((SIZE - row.size) / 2), int((SIZE - row.size) / 2)), 'constant', constant_values=0)
        label = np.pad(label, (int((SIZE - label.size) / 2), int((SIZE - label.size) / 2)), 'constant', constant_values=0)
        label = tf.one_hot(label, NUM_OF_CLASSES)
        return (row, label)
    
    def return_all_data(self):
        data = []
        labels = []
        for i in range(len(self)):
            row, label = self[i]
            data.append(row)
            labels.append(label)
        return data, labels
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])