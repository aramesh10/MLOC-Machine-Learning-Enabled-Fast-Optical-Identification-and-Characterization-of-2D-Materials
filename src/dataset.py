import os
import matplotlib.image as img

import numpy as np
import torch
from torch.utils.data import Dataset

class GrapheneDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, label_transform=None):
        self.img = torch.tensor(np.array([img.imread(img_dir+image) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)]))               #  if os.path.isfile(img_dir+img)
        self.label = torch.tensor(np.array([img.imread(label_dir+label) for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)]))       #  if os.path.isfile(label_dir+label)
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
        return img, label