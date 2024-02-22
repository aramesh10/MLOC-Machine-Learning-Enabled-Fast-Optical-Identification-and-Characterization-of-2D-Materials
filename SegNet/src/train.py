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

train_img_dir = './data/train/images/'
train_label_dir = './data/train/labels/'
test_img_dir = './data/test/'
test_label_dir = './data/test/label/'
model_dir = './model/'

def load(model, weight_fn):
    assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)

    checkpoint = torch.load(weight_fn)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("Checkpoint is loaded at {} | Epochs: {}".format(weight_fn, epoch))

if __name__ == '__main__':
    # Train model
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = GrapheneDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform, label_transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    SegNet.Train.Train(train_dataloader, None)

    # Test model
    test_data = GrapheneDataset(img_dir=test_img_dir, label_dir=None, transform=transform, label_transform=transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = SegNet.SegNet()
    model.eval()
    load(model, 'segnet_weights.pth.tar')

    for i, data in enumerate(test_dataloader):
        image = data[0]
        res = model(image).detach().numpy().reshape((100,100,3))

        print(res)

        plt.figure()
        plt.imshow(image.detach().numpy().reshape((100,100,3)))
        plt.figure()
        plt.imshow(np.round(res))
        plt.show()