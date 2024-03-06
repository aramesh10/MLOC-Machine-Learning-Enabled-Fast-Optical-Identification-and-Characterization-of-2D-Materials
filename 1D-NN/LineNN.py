import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

from model import OneDim_CNN
from dataset import GrapheneDataset

BATCH_SIZE = 1

if __name__ == '__main__':
    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'

    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, preprocess_bool=True)
    val_graphene_dataset   = GrapheneDataset(x_val_dir, y_val_dir, preprocess_bool=True)
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=True)

    train_graphene_dataloader = DataLoader(train_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_graphene_dataloader = DataLoader(val_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_graphene_dataloader = DataLoader(test_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = OneDim_CNN()
    model.train()