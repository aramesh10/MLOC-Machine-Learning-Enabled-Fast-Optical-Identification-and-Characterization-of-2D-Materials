import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from torchvision.io import read_image, ImageReadMode

import os
import torch
from torch.utils.data import DataLoader
import SegNet
from dataset import GrapheneDataset

BATCH_SIZE = 1

train_img_dir = './data/images/'
train_label_dir = './data/labels/'
if __name__ == '__main__':
    train_data = GrapheneDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=None, label_transform=None)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    train_image = train_features[0].squeeze()
    train_label = train_labels[0]
    plt.figure()
    plt.imshow(train_image)
    plt.figure()
    plt.imshow(train_label)
    plt.show()