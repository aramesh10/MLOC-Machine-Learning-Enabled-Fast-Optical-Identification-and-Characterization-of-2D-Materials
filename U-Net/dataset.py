import os
import numpy as np
import cv2
import matplotlib
import matplotlib.image as img
import matplotlib.pyplot as plt
import albumentations as albu

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NUM_OF_CLASSES = 3
SIZE = 32*3
class GrapheneDataset(Dataset):
    
    def __init__(self, img_dir, label_dir, preprocess_bool=False, img_format=None, normalize=False):
        assert(np.any(['plasma', 'hsv', 'spectral'] == img_format), "img_format must be one of the following: ['plasma', 'hsv', 'spectral']")

        self.img = np.array([img.imread(img_dir+image) for image in os.listdir(img_dir) if os.path.isfile(img_dir+image)])  
        self.img = self.img[:, :SIZE, :SIZE]

        if label_dir != None:
            self.label = np.array([img.imread(label_dir+label) for label in os.listdir(label_dir) if os.path.isfile(label_dir+label)])
            self.label = self.label[:, :SIZE, :SIZE]
            assert(len(self.img) == len(self.label))
        else:
            self.label = None
        
        self.preprocess_bool = preprocess_bool
        if preprocess_bool:
            self.transform = self.preprocessing()
        else:
            self.transform = None

        self.img_format = img_format
        self.normalize = normalize

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.label[idx]

        # Image transformations
        if self.preprocess_bool:
            orig_image = image

            image_temp = image
            image_temp = rgb2gray(image_temp)
            image_temp = self.transform(image=image_temp)["image"]
            image_temp = self.image_normalize(image=image_temp)

            image = convertImage(image, self.img_format)
            if normalize:
                image = rgb2gray(image)
            image = self.transform(image=image)["image"]
            image = self.image_normalize(image=image)
            
            IMAGE_WEIGHT = 1
            image_sum = IMAGE_WEIGHT*image + (1-IMAGE_WEIGHT) * image_temp
            image_sum = self.image_normalize(image_sum)
            
            # show_preprocessing(orig_image, label, image_temp, image, image_sum)
        else:
            image = rgb2gray(image)
        image = self.image_to_tensor(image=image)

        # Label transformations
        label = self.label_to_tensor(label=label)

        return (image, label)

    def preprocessing(self):
        transform = [
            albu.Sharpen(alpha=(0,1), p=1)
        ]
        return albu.Compose(transform)

    def image_to_tensor(self, image):
        return torch.tensor(image).float().reshape(1, SIZE, SIZE)

    def image_normalize(self, image):
        # shape = image.shape
        img_norm = normalize(image)#.reshape(shape)
        return img_norm

    def label_to_tensor(self, label):
        return torch.tensor(255*label-1).reshape(1, SIZE, SIZE).to(torch.int64)

def show_preprocessing(orig_image, label, image_temp, image, image_sum):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)   
    fig.suptitle('Image Preprocessing')
    ax1.imshow(np.array(orig_image))
    ax1.set_title("Original Image")
    ax2.imshow(np.array(label))
    ax2.set_title("Label")
    ax3.imshow(np.array(image_temp))
    ax3.set_title("Previous Preprocessing")
    ax4.imshow(np.array(image))
    ax4.set_title("New Preprocessing")
    ax5.imshow(np.array(image_sum))
    ax5.set_title("Sum Preprocessing")
    plt.show()

def post_process_mask_prediction(prediction_mask):
    post_process_mask = reorder_arr(np.squeeze(prediction_mask), (96, 96, 3))
    post_process_mask = reverse_one_hot(post_process_mask)
    return post_process_mask
    
def reorder_arr(arr, shape):
    assert(len(shape) == 3)
    reorder_arr = []
    for i in range(shape[0]):
        temp1 = []
        for j in range(shape[1]):
            append_arr = [arr[0][i][j], arr[1][i][j], arr[2][i][j]]
            temp1.append(append_arr)
        reorder_arr.append(temp1)
    reorder_arr = np.array(reorder_arr)
    return reorder_arr

def reverse_one_hot(arr):
    arr_shape = arr.shape
    reverse_one_hot_arr = np.empty((arr_shape[0], arr_shape[1]))
    for row in range(arr_shape[0]):
        for col in range(arr_shape[1]):
            reverse_one_hot_arr[row][col] = np.argmax(arr[row][col])
    return reverse_one_hot_arr

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def OLD_convertImage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                    # Convert the image to grayscale
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)                 # Apply Gaussian blur to the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))             # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    enhanced_image = clahe.apply(blurred_image)
    hist = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])       # Compute histogram for the enhanced image

    # Find peaks in the histogram & Calculate threshold values
    peaks = np.where(hist > np.max(hist) * 0.1)[0]      # Adjust the threshold (0.1) as needed
    low_intensity = np.min(peaks)
    high_intensity = np.max(peaks)

    # Define medium intensity levels (you may adjust these based on your specific requirements)
    medium_intensity_1 = low_intensity + (high_intensity - low_intensity) // 4
    medium_intensity_2 = low_intensity + (high_intensity - low_intensity) // 2
    medium_intensity_3 = low_intensity + 3 * (high_intensity - low_intensity) // 4

    # Define a color mapping function based on intensity
    # Map intensity levels to colors based on low, medium, and high intensity ranges
    def map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity):
        if intensity <= low_intensity:
            return (153, 153, 255)  # Blue for low intensity
        elif intensity < medium_intensity_1:
            return (150, 255, 150)  # Green for medium intensity 1
        elif intensity < medium_intensity_2:
            return (255, 150, 150)  # Red for medium intensity 2
        elif intensity < medium_intensity_3:
            return (150, 255, 255)  # Yellow for medium intensity 3
        else:
            return (255, 255, 255)  # White for high intensity

    # Apply color mapping to each pixel of the enhanced grayscale image
    colorized_image = np.zeros((enhanced_image.shape[0], enhanced_image.shape[1], 3), dtype=np.uint8)
    for y in range(enhanced_image.shape[0]):
        for x in range(enhanced_image.shape[1]):
            intensity = enhanced_image[y, x]
            color = map_intensity_to_color(intensity, low_intensity, medium_intensity_1, medium_intensity_2, medium_intensity_3, high_intensity)
            colorized_image[y, x] = color

    return colorized_image

def convertImage(image, img_format):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # convert to l*a*b colorspace

    # measure substrate ~dynamically~
    region_percent = 0.2  # set row based on region % of image, 0.2 =  measure 20% from top
    row = int(lab_image.shape[0] * region_percent)

    # measure l*a*b through given region
    horizontal_line = []
    for col in range(lab_image.shape[1]):
        lab_values = lab_image[row, col]
        horizontal_line.append(lab_values)

        substrate_lab = np.mean(horizontal_line, axis=0)   # l*a*b mean of substrate

        # compute delta E for each pixel relative to substrate
        delta_e = np.linalg.norm(lab_image - substrate_lab, axis=2)
        normalized_delta_e = delta_e / np.max(delta_e) #important for visualization

        # Conversions
        # Perceptually Uniform Sequential Colormap
        if img_format == 'plasma':
            colormap = matplotlib.colormaps['plasma']
            normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
            bgr_plasma = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            return bgr_plasma

        # hsv colormap
        elif img_format == 'hsv':
            colormap = matplotlib.colormaps['hsv']
            normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
            bgr_hsv = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            return bgr_hsv

        # Diverging Colormap
        elif img_format == 'spectral':
            colormap = matplotlib.colormaps['Spectral']
            normalized_delta_e_rgb = colormap(normalized_delta_e)[:, :, :3]
            bgr_spectral = cv2.cvtColor((normalized_delta_e_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            return bgr_spectral

def calc_metrics(prediction, label):
    pred = np.array(prediction).reshape(96, 96)
    y = np.array(label).reshape(96, 96)
    overall_accuracy = (np.sum(pred == y)) / (pred.shape[0] * pred.shape[1])
    confusion_mat = confusion_matrix(pred.flatten(), y.flatten(), labels=[0, 1, 2])
    precision, recall, f1_score, support = precision_recall_fscore_support(y.flatten(), pred.flatten(), warn_for=[])
    label_acc = confusion_mat.diagonal()/confusion_mat.sum(axis=1)
    return overall_accuracy, label_acc, confusion_mat, precision, recall, f1_score 