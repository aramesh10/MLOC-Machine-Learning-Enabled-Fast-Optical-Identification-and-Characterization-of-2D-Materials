import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fpn_model import FPN
from dataset_tf import GrapheneDataset

if __name__ == '__main__':
    # Best Config

    # Model Configurations
    signal_length = 100  # Length of each Segment
    model_depth = 3  # Number of Level in the CNN Model
    model_width = 10  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 15  # Size of the Kernels/Filter
    num_channel = 1  # Number of Channels in the Model
    D_S = 0  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 0  # Turn on Guided Attention
    problem_type = 'Classification'
    output_nums = 3  # Number of Class for Classification Problems, always '1' for Regression Problems
    feature_number = 1024  # Number of Features to be Extracted, only required if the AutoEncoder Mode is turned on
    model_name = 'FPN'  # FPN

    Model = FPN(signal_length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, is_transconv=False).FPN()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()

    model_name_file = './fpn_model.keras' # CHANGE FOR DIFFERENT MODELS

    # Dataset Dir
    use_filtered_data = True
    if use_filtered_data:
        x_train_dir = './data/filtered_dataset/train/images/'
        y_train_dir = './data/filtered_dataset/train/labels/'
        x_val_dir = './data/filtered_dataset/val/images/'
        y_val_dir = './data/filtered_dataset/val/labels/'
        x_test_dir = './data/filtered_dataset/test/images/'
        y_test_dir = './data/filtered_dataset/test/labels/'
    else:
        x_train_dir = './data/train/images/'
        y_train_dir = './data/train/labels/'
        x_val_dir = './data/val/images/'
        y_val_dir = './data/val/labels/'
        x_test_dir = './data/test/images/'
        y_test_dir = './data/test/labels/'

    # Model Implementation
    loss = tf.keras.losses.CategoricalCrossentropy()               # tf.keras.losses.CategoricalCrossentropy() # DiceLoss() # tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=15)
    metric = tf.keras.metrics.CategoricalCrossentropy()            # tf.keras.metrics.CategoricalCrossentropy()
    Model = FPN(signal_length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums, ds=D_S, ae=A_E, ag=A_G, is_transconv=False).FPN()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
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
                tf.keras.callbacks.ModelCheckpoint(model_name_file, verbose=1, monitor='val_loss', save_best_only=True, mode='min')]
    history = Model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=1, validation_data= (x_val, y_val), shuffle=True, callbacks=callbacks)

    # Test
    Model.load_weights(model_name_file) 

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
