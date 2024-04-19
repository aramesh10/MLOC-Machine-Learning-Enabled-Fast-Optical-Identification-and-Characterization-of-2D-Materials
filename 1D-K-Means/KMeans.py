import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
from torch import nn
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == '__main__':
    x_train_dir = './data/filtered_dataset/train/images/'
    y_train_dir = './data/filtered_dataset/train/labels/'

    # Get data
    x_train = np.array([rgb2gray(img.imread(x_train_dir+image)) for image in os.listdir(x_train_dir) if os.path.isfile(x_train_dir+image)])     # Shape: [N, W, H]
    y_train = np.array([img.imread(y_train_dir+label)*255 for label in os.listdir(y_train_dir) if os.path.isfile(y_train_dir+label)])           # Shape: [N, W, H]

    # Segment data into rows
    x_train_row = np.reshape(x_train, (x_train.shape[0]*x_train.shape[2], x_train.shape[1])) # Shape: [N*H, W]
    y_train_row = np.reshape(y_train, (y_train.shape[0]*y_train.shape[2], y_train.shape[1])) # Shape: [N*H, W]

    ROW = 20
    for n in range(100):
        n = np.random.choice(x_train.shape[0])

        # fig, axes= plt.subplot_mosaic("ABEEE;CCEEE;DDEEE")
        fig, axes= plt.subplot_mosaic("ACC;XDD;BEE")
        plt.title("Pixel Intensity vs. Label")
        axes['A'].imshow(x_train[n])
        axes['A'].set_title("Image")
        axes['B'].imshow(y_train[n])
        axes['B'].set_title("Labels")
        axes['C'].plot(range(x_train_row.shape[1]), x_train_row[(100*n)+ROW], label='Pixel Intensity')
        axes['C'].set_title(f"Row {ROW} values")
        axes['C'].set_ylabel("Intensity")
        axes['D'].plot(range(y_train_row.shape[1]), y_train_row[(100*n)+ROW], label='Label')
        axes['D'].set_title(f"Row {ROW} labels")
        axes['D'].set_ylabel("Label")

        x_train_norm = normalize(np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2])))
        x_train_all  = normalize(x_train[1].flatten().reshape(1, -1))
        y_train_all  = y_train[1].flatten()

        axes['E'].set_title('K-Means Clusters')
        axes['E'].plot(x_train_all.reshape(-1,1), y_train_all, 'bo')
        axes['E'].set_xlabel('Normalized pixel value')
        axes['E'].set_ylabel('Label')
        axes['X'].axis('off')
        plt.show()
