import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import OneDim_CNN, OneDim_CNN_2
from dataset import GrapheneDataset

EPOCHS = 10
BATCH_SIZE = 1

if __name__ == '__main__':
    # Init data
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
    test_graphene_dataloader = DataLoader(test_graphene_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = OneDim_CNN_2()                                                        # Model
    loss_fn = torch.nn.CrossEntropyLoss()                                       # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)     # Optimizer

    # def train_one_epoch(training_loader, epoch_index):
    #     running_loss = 0.
    #     last_loss = 0.

    #     # Here, we use enumerate(training_loader) instead of
    #     # iter(training_loader) so that we can track the batch
    #     # index and do some intra-epoch reporting
    #     for i, data in tqdm(enumerate(training_loader)):
    #         inputs, labels = data       # Every data instance is an input + label pair 

    #         optimizer.zero_grad()       # Zero your gradients for every batch!
    #         outputs = model(inputs)     # Make predictions for this batch

    #         # Compute the loss and its gradients
    #         loss = loss_fn(outputs, labels)
    #         loss.backward()

    #         # Adjust learning weights
    #         optimizer.step()

    #         # Gather data and report
    #         running_loss += loss.item()
    #         if i % BATCH_SIZE == BATCH_SIZE-1:
    #             last_loss = running_loss / 1000 # loss per batch
    #             # print('  batch {} loss: {}'.format(i + 1, last_loss))
    #             running_loss = 0.

    #     return last_loss

    # # Initializing in a separate cell so we can easily add more epochs to the same run
    # # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # epoch_number = 0
    # best_vloss = 1_000_000.
    # for epoch in range(EPOCHS):
    #     print('EPOCH {}:'.format(epoch_number + 1))

    #     # Make sure gradient tracking is on, and do a pass over the data
    #     model.train(True)
    #     avg_loss = train_one_epoch(train_graphene_dataloader, epoch_number)

    #     running_vloss = 0.0
    #     # Set the model to evaluation mode, disabling dropout and using population
    #     # statistics for batch normalization.
    #     model.eval()

    #     # Disable gradient computation and reduce memory consumption.
    #     with torch.no_grad():
    #         for i, vdata in enumerate(val_graphene_dataloader):
    #             vinputs, vlabels = vdata
    #             voutputs = model(vinputs)
                
    #             vloss = loss_fn(voutputs, vlabels)
    #             running_vloss += vloss

    #     avg_vloss = running_vloss / (i + 1)
    #     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    #     # Track best performance, and save the model's state
    #     if avg_vloss < best_vloss:
    #         best_vloss = avg_vloss
    #         model_path = 'best_model_preprocessing'
    #         torch.save(model.state_dict(), model_path)

    #     epoch_number += 1

    saved_model = OneDim_CNN_2()
    saved_model.load_state_dict(torch.load('best_model_preprocessing'))

    image = []
    label = []
    pred = []
    with torch.no_grad():
        for i, row in enumerate(test_graphene_dataloader):
            tinputs, tlabels = row
            toutputs = model(tinputs)

            image.append(tinputs)
            label.append(tlabels)
            pred.append(toutputs)

            if (i%100 == 99):
                image = np.array(image).squeeze()
                label = np.array(label).squeeze()
                pred = np.array(pred).squeeze()
                pred = np.argmax(pred, axis=1).squeeze()

                # print(image)
                # print(image.shape)
                # print(label)
                # print(label.shape)
                print(pred)
                print(pred.shape)

                plt.figure()
                plt.imshow(image)
                plt.figure()
                plt.imshow(label)
                plt.figure()
                plt.imshow(pred)
                plt.show()

                image = []
                label = []
                pred = []