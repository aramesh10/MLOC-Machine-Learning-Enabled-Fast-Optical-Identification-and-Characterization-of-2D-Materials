import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model_tf import UNet
from dataset_tf import GrapheneDataset

if __name__ == '__main__':
    # Model Configurations
    length = 128                   # Length of each Segment of a 1D Signal
    model_name = 'UNet'             # Name of the Segmentation Model
    model_depth = 1                 # Number of Level in the CNN Model
    model_width = 4                # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3                 # Size of the Kernels/Filter
    num_channel = 1                 # Number of Channels in the Model
    D_S = 1                         # Turn on Deep Supervision
    A_E = 0                         # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 1                         # Turn on for Guided Attention (Creates 'Attention Guided UNet')
    LSTM = 1                        # Turn on for LSTM (Creates BCD-UNet)
    problem_type = 'Classification'     # Regression or Classification (Commonly Regression)
    output_nums = 3                 # Number of Classes for Classification Problems, always '1' for Regression Problems
    is_transconv = True             # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024           # Number of Features to be Extracted

    # Dataset Dir
    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'

    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    # Model Implementation
    Model = UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                 ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, is_transconv=is_transconv).UNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=tf.keras.metrics.CategoricalCrossentropy())
    # Model.summary()

    # Dataset
    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, preprocess_bool=False)
    val_graphene_dataset   = GrapheneDataset(x_val_dir, y_val_dir, preprocess_bool=False)
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=False)

    x_train, y_train = train_graphene_dataset.return_all_data()
    x_val, y_val = val_graphene_dataset.return_all_data()
    x_test, y_test = test_graphene_dataset.return_all_data()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Train
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min'), 
                tf.keras.callbacks.ModelCheckpoint('best_model_tf.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='min')]
    history = Model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data= (x_val, y_val), shuffle=True, callbacks=callbacks)
