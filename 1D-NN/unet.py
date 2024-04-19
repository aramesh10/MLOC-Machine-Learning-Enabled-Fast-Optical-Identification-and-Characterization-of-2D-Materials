import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from unet_model import UNet, DiceLoss
from dataset_tf import GrapheneDataset, calc_metrics

TRAIN = True
TEST = True
RUN_NAME = 'new_preprocessing_spectral_normalize'   # CHANGE FOR DIFFERENT MODELS
USE_AUGMENTED_DATA = True
PREPROCESS_BOOL = True
IMAGE_FORMAT = 'spectral'     # ['plasma', 'hsv', 'spectral']
GRAYSCALE_BOOL = True
NORMALIZE_BOOL = True
if __name__ == '__main__':
    # Model Configurations
    model_name_file = f"./1D-U-Net/runs/{RUN_NAME}/{RUN_NAME}.keras" 
    length = 100                        # Length of each Segment of a 1D Signal
    model_name = 'UNet'                 # Name of the Segmentation Model
    model_depth = 1                     # Number of Level in the CNN Model
    model_width = 4                     # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 15                    # Size of the Kernels/Filter
    num_channel = 1                     # Number of Channels in the Model
    D_S = 0                             # Turn on Deep Supervision
    A_E = 0                             # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 1                             # Turn on for Guided Attention (Creates 'Attention Guided UNet')
    LSTM = 1                            # Turn on for LSTM (Creates BCD-UNet)
    problem_type = 'Classification'     # Regression or Classification (Commonly Regression)
    output_nums = 3                     # Number of Classes for Classification Problems, always '1' for Regression Problems
    is_transconv = True                 # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024               # Number of Features to be Extracted

    # Create directory if not exist
    if not os.path.exists(f"./1D-U-Net/runs/{RUN_NAME}"):
        os.makedirs(f"./1D-U-Net/runs/{RUN_NAME}")
        os.makedirs(f"./1D-U-Net/runs/{RUN_NAME}/figures")
        print(f"Created ./1D-U-Net/runs/{RUN_NAME} directory")

    # Dataset Dir
    use_filtered_data = not USE_AUGMENTED_DATA
    if use_filtered_data:
        x_train_dir = '../data/filtered_dataset/train/images/'
        y_train_dir = '../data/filtered_dataset/train/labels/'
        x_val_dir = '../data/filtered_dataset/val/images/'
        y_val_dir = '../data/filtered_dataset/val/labels/'
        x_test_dir = '../data/filtered_dataset/test/images/'
        y_test_dir = '../data/filtered_dataset/test/labels/'
    else:
        x_train_dir = '../data/train/images/'
        y_train_dir = '../data/train/labels/'
        x_val_dir = '../data/val/images/'
        y_val_dir = '../data/val/labels/'
        x_test_dir = '../data/test/images/'
        y_test_dir = '../data/test/labels/'

    # Model Implementation
    loss = tf.keras.losses.CategoricalCrossentropy()       # tf.keras.losses.CategoricalCrossentropy() # DiceLoss() # tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=15)
    metric = tf.keras.metrics.CategoricalCrossentropy()            # tf.keras.metrics.CategoricalCrossentropy()
    Model = UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                 ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, is_transconv=is_transconv).UNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=loss, metrics=metric)
    Model.summary()

    # Dataset
    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, preprocess_bool=PREPROCESS_BOOL, img_format=IMAGE_FORMAT, grayscale=GRAYSCALE_BOOL, normalize=NORMALIZE_BOOL)
    val_graphene_dataset   = GrapheneDataset(x_val_dir, y_val_dir, preprocess_bool=PREPROCESS_BOOL, img_format=IMAGE_FORMAT, grayscale=GRAYSCALE_BOOL, normalize=NORMALIZE_BOOL)
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=PREPROCESS_BOOL, img_format=IMAGE_FORMAT, grayscale=GRAYSCALE_BOOL, normalize=NORMALIZE_BOOL)

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
    if TRAIN:
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min'), 
                    tf.keras.callbacks.ModelCheckpoint(model_name_file, verbose=1, monitor='val_loss', save_best_only=True, mode='min')]
        history = Model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=1, validation_data= (x_val, y_val), shuffle=True, callbacks=callbacks)

    # Test
    if TEST:
        Model.load_weights(model_name_file) 
        test_data_arr = []
        prediction_arr = []
        label_data_arr = []
        for i, (image, label) in enumerate(zip(x_test, y_test)):
            image = image.reshape(1, -1, 1)
            pred = Model.predict(image, verbose=0) 
            test_data_arr.append(image.reshape(-1))
            label_data_arr.append(np.argmax(label, axis=1).reshape(-1))
            prediction_arr.append(np.argmax(pred, axis=2).reshape(-1))

            if (i+1)%100 == 0:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                fig.suptitle('Model Training Results')
                ax1.imshow(np.array(test_data_arr))
                ax1.set_title("Original Image")
                ax2.imshow(np.array(label_data_arr))
                ax2.set_title("Labels")
                ax3.imshow(np.array(prediction_arr))
                ax3.set_title("Prediction")
                plt.savefig(f"./1D-U-Net/runs/{RUN_NAME}/figures/example_{int((i+1)/100)}.png")
                test_data_arr = []
                label_data_arr = []
                prediction_arr = []

            if (i+1) == 500:
                break;

        overall_accuracy_arr = np.empty(0)
        label_accuracy_arr = np.empty((0,3))
        precision_arr = np.empty(0)
        recall_arr = np.empty(0)
        f1_arr = np.empty(0)
        confusion_matrix = np.zeros((3,3))
        test_data_arr = []
        prediction_arr = []
        label_data_arr = []
        for i, (test_data, label_data) in tqdm(enumerate(zip(x_test, y_test))):
            test_data = test_data.reshape(1, -1, 1)
            pred = Model.predict(test_data, verbose=0) 

            test_data_arr.append(test_data.reshape(-1))
            label_data_arr.append(np.argmax(label_data, axis=1).reshape(-1))
            prediction_arr.append(np.argmax(pred, axis=2).reshape(-1))

            if (i+1)%100 == 0:
                overall_accuracy, label_acc, confusion_mat, precision, recall, f1_score = calc_metrics(prediction_arr, label_data_arr)
                overall_accuracy_arr = np.append(overall_accuracy_arr, overall_accuracy)
                label_accuracy_arr = np.append(label_accuracy_arr, label_acc.reshape(1, -1), axis=0)
                precision_arr = np.append(precision_arr, precision)
                recall_arr = np.append(recall_arr, recall)
                f1_arr = np.append(f1_arr, f1_score)
                confusion_matrix += confusion_mat
                test_data_arr = []
                label_data_arr = []
                prediction_arr = []

        print(label_accuracy_arr)

        # Print metric
        print("*"*50)
        print("Precision:", np.mean(precision_arr))
        print("Recall:", np.mean(recall_arr))
        print("F1 Score:", np.mean(f1_arr))
        print()
        print("Confusion Matrix")
        print(f"{confusion_matrix}")
        print()
        print("Label Accuracies")
        print("\tSubstrate:", np.nanmean(label_accuracy_arr[:, 0]))
        print("\tMonolayer:", np.nanmean(label_accuracy_arr[:, 1]))
        print("\tMulti-Layer:", np.nanmean(label_accuracy_arr[:, 2]))
        print()
        print("OVERALL ACCURACY:", np.nanmean(overall_accuracy_arr))
        print("*"*50)

        # Write results to file
        with open(f"./1D-U-Net/runs/{RUN_NAME}/metrics.txt", "w+") as metric_file:
            metric_file.write(f"Precision: {np.nanmean(precision_arr)}\n")
            metric_file.write(f"Recall: {np.nanmean(recall_arr)}\n")
            metric_file.write(f"F1 Score: {np.nanmean(f1_arr)}\n")
            metric_file.write(f"\n")
            metric_file.write(f"Confusion Matrix\n")
            metric_file.write(f"{confusion_matrix}\n")
            metric_file.write(f"\n")
            metric_file.write(f"Label Accuracies\n")
            metric_file.write(f"\tSubstrate: {np.nanmean(label_accuracy_arr[:, 0])}\n")
            metric_file.write(f"\tMonolayer: {np.nanmean(label_accuracy_arr[:, 1])}\n")
            metric_file.write(f"\tMulti-Layer: {np.nanmean(label_accuracy_arr[:, 2])}\n")
            metric_file.write(f"\n\n")
            metric_file.write(f"OVERALL ACCURACY: {np.nanmean(overall_accuracy_arr)}")
            print(f"Wrote to ./1D-U-Net/runs/{RUN_NAME}/metrics.txt")
