import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
from torch import nn
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

""" TODO: Total Color Difference (TCD) """

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == '__main__':
    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'

    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    # Get data
    x_train = np.array([rgb2gray(img.imread(x_train_dir+image)) for image in os.listdir(x_train_dir) if os.path.isfile(x_train_dir+image)])     # Shape: [N, W, H]
    y_train = np.array([img.imread(y_train_dir+label)*255 for label in os.listdir(y_train_dir) if os.path.isfile(y_train_dir+label)])           # Shape: [N, W, H]

    # Segment data into rows
    x_train_row = np.reshape(x_train, (x_train.shape[0]*x_train.shape[2], x_train.shape[1])) # Shape: [N*H, W]
    y_train_row = np.reshape(y_train, (y_train.shape[0]*y_train.shape[2], y_train.shape[1])) # Shape: [N*H, W]

    plt.figure()
    plt.imshow(x_train[1])
    plt.figure()
    plt.imshow(y_train[1])

    plt.figure()
    plt.title("Pixel Intensity vs. Label")
    plt.subplot(211)
    plt.plot(range(x_train_row.shape[1]), x_train_row[180], label='Pixel Intensity')
    plt.subplot(212)
    plt.plot(range(y_train_row.shape[1]), y_train_row[180], label='Label')
    plt.legend()
    plt.show()

    x_train_norm = normalize(np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2])))
    x_train_all = normalize(x_train[1].flatten().reshape(1, -1))
    y_train_all = y_train[1].flatten()

    plt.plot(x_train_all.reshape(-1,1), y_train_all, 'bo')
    plt.show()

    # kmeans_model = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)