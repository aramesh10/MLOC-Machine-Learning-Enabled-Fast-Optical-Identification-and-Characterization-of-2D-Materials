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
    model_depth = 3                 # Number of Level in the CNN Model
    model_width = 7                # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 21                 # Size of the Kernels/Filter
    num_channel = 1                 # Number of Channels in the Model
    D_S = 0                         # Turn on Deep Supervision
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
    Model.summary()

    # Dataset
    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, preprocess_bool=True)
    val_graphene_dataset   = GrapheneDataset(x_val_dir, y_val_dir, preprocess_bool=True)
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=True)

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
    history = Model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=1, validation_data= (x_val, y_val), shuffle=True, callbacks=callbacks)

    # Test
    # saved_model = tf.keras.models.load_model('best_model_tf.h5')
    Model.load_weights('best_model_tf.h5')
    
    # data = []
    # for n, train_data in enumerate(x_train):
    #     print(train_data.shape)
    #     print(train_data)
    #     data.append(train_data)
    #     if (n+1) % 100 == 0:
    #         plt.figure()
    #         plt.imshow(data)
    #         plt.show()
    #         data = []

    test_data_arr = []
    prediction_arr = []
    label_data_arr = []
    for i, (test_data, label_data) in enumerate(zip(x_test, y_test)):
        test_data = test_data.reshape(1, -1, 1)
        pred = Model.predict(test_data) 

        test_data_arr.append(test_data.reshape(-1))
        label_data_arr.append(np.argmax(label_data, axis=1).reshape(-1))
        prediction_arr.append(np.argmax(pred, axis=2).reshape(-1))

        if (i+1)%100 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.set_title('Original Image')
            ax1.imshow(test_data_arr)
            ax2 = fig.add_subplot(132)
            ax2.set_title('Labels')
            ax2.imshow(label_data_arr, vmin=0, vmax=2)
            ax3 = fig.add_subplot(133)
            ax3.set_title('Prediction')
            ax3.imshow(prediction_arr, vmin=0, vmax=2)
            plt.show()

            test_data_arr = []
            label_data_arr = []
            prediction_arr = []
