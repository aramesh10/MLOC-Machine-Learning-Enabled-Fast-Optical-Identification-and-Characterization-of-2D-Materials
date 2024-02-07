import os
import matplotlib.image as img

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

NUM_OF_CLASSES = 3
class GrapheneDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, label_transform=None):
        self.img = np.array([img.imread(img_dir+image) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])            #  if os.path.isfile(img_dir+img)
        self.label = np.array([img.imread(label_dir+label) for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])     #  if os.path.isfile(label_dir+label)
        self.transform = transform
        self.label_transform = label_transform
        assert(len(self.img) == len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        if self.transform != None:
            img = self.transform(img)
        if self.label_transform != None:
            label = self.label_transform(label)
        img = img.reshape(3, 100, 100)
        label = 255*label.reshape(1, 100, 100)-1
        label = one_hot(label.to(torch.int64), 3).reshape(3, 100, 100).float()
        return (img, label) # change output class
    