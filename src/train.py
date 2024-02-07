import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import SegNet
from dataset import GrapheneDataset

from torch.nn.functional import one_hot


import matplotlib.image as img

BATCH_SIZE = 2

train_img_dir = './data/images/'
train_label_dir = './data/labels/'
model_dir = './model/'
if __name__ == '__main__':
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_data = GrapheneDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform, label_transform=transform)
    # train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    # train_image, train_label = next(iter(train_dataloader))

    # train_label_one_hot = one_hot(train_label.to(torch.int64), 3)

    # print("TRAIN LABELS")
    # print(train_label)
    # print(f"train label shape: {train_label.size()}")

    # print("\n"+ "*"*50)

    # print("TRAIN LABELS ONE HOT")
    # print(train_label_one_hot)
    # print(f"train label shape: {train_label_one_hot.size()}")

    # plt.figure()
    # plt.imshow(train_image.reshape(100, 100, 3))
    # plt.figure()
    # plt.imshow(train_label.reshape(100, 100, 1))
    # plt.figure()
    # plt.imshow(train_label_one_hot.reshape(100, 100, 3))
    # plt.show()
    
    # ---
    
    # files = os.listdir(train_img_dir)
    # data_list = []
    # for image in files:
    #     if os.path.isfile(train_img_dir+image):
    #         print(image)
    #         image_f = torch.tensor(img.imread(train_img_dir+image)).float().reshape(100, 100, 3)
    #         data_list.append(image_f)
    # data = torch.tensor(data_list)
    
    # --- 

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = GrapheneDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform, label_transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.shape)
    
    SegNet.Train.Train(train_dataloader, None)

    # --- 

    # # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # train_image = train_features[0].squeeze()
    # train_label = train_labels[0] * 255
    # plt.figure()
    # plt.imshow(train_image)
    # plt.figure()
    # plt.imshow(train_label)
    # plt.show()