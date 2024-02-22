import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
v 
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# class GrapheneDataset:
#     """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline 
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing 
#             (e.g. noralization, shape manipulation, etc.)
    
#     """
    
#     CLASSES = ['substrate', 'mono-layer', 'multi-layer']
    
#     def __init__(
#             self, 
#             images_dir, 
#             masks_dir, 
#             classes=None, 
#             augmentation=None, 
#             preprocessing=None,
#     ):
#         self.ids_img = os.listdir(images_dir)
#         self.ids_label = os.listdir(masks_dir)
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_img]
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_label]
        
#         # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
    
#     def __getitem__(self, i):
#         # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:96, :96].reshape(3, 96, 96)
#         mask = cv2.imread(self.masks_fps[i], 0)[:96, :96]

#         print('*'*50)
#         print(image.shape)
#         print(mask.shape)
#         print('*'*50)

#         # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype('uint8') # .astype('float')

#         print(mask.shape)
#         print(type(mask[0]))
#         print(type(mask[0][0]))
#         print(type(mask[0][0][0]))
#         print(mask)
#         print('*'*50)
        
#         # add background if mask is not binary
#         if mask.shape[-1] != 1:
#             background = 1 - mask.sum(axis=-1, keepdims=True)
#             mask = np.concatenate((mask, background), axis=-1)
        
#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
        
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']

#         print(torch.tensor(image.astype('float32')).shape)

#         return torch.tensor(image.astype('float32')), torch.tensor(mask.astype('float32'))
        
#     def __len__(self):
#         return len(self.ids_img)

class GrapheneDataloader(DataLoader):
    """Load data from graphene dataset and form batches
    # 
    Args:
        dataset: instance of Graphene Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()
 
    def __getitem__(self, i): 
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)] 
        return batch
 
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
 
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)  
    
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
    
    def __len__(self):
        return len(self.ids)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20

if __name__ == '__main__':
    x_all_data_dir = './data/all_data/images/'
    y_all_data_dir = './data/all_data/labels/'

    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'

    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    # x_all_data_files = os.listdir(x_all_data_dir)
    # y_all_data_files = os.listdir(y_all_data_dir)
    
    # x_train_files, x_test_files, y_train_files, y_test_files = train_test_split(x_all_data_files, y_all_data_files, random_state=1, train_size=TRAIN_SPLIT, shuffle=True) 
    # x_train_files, x_val_files, y_train_files, y_val_files = train_test_split(x_train_files, y_train_files, random_state=1, train_size=TRAIN_SPLIT, shuffle=True) 
    
    # # Train
    # for image, mask in tqdm(zip(x_train_files, y_train_files)):
    #     try:
    #         shutil.copyfile(x_all_data_dir+image, x_train_dir+image)
    #         shutil.copyfile(y_all_data_dir+mask, y_train_dir+mask)
    #     except:
    #         print("Train Error")
    #         print("Image =", image)
    #         print("mask =", mask)
    #         print()

    # # Val
    # for image, mask in tqdm(zip(x_val_files, y_val_files)):
    #     try:
    #         shutil.copyfile(x_all_data_dir+image, x_val_dir+image)
    #         shutil.copyfile(y_all_data_dir+mask, y_val_dir+mask)
    #     except:
    #         print("Val Error")
    #         print("Image =", image)
    #         print("mask =", mask)
    #         print()

    # # Test
    # for image, mask in tqdm(zip(x_test_files, y_test_files)):
    #     try:
    #         shutil.copyfile(x_all_data_dir+image, x_test_dir+image)
    #         shutil.copyfile(y_all_data_dir+mask, y_test_dir+mask)
    #     except:
    #         print("Test Error")
    #         print("Image =", image)
    #         print("mask =", mask)
    #         print()

    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])
    val_graphene_dataset = GrapheneDataset(x_val_dir, y_val_dir, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])
    test_graphene_dataset = GrapheneDataset(x_test_dir, y_test_dir, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])

    train_graphene_dataloader = GrapheneDataloader(train_graphene_dataset, batch_size=1, shuffle=True)
    val_graphene_dataloader = GrapheneDataloader(val_graphene_dataset, batch_size=1, shuffle=True)
    test_graphene_dataloader = GrapheneDataloader(test_graphene_dataset, batch_size=1, shuffle=True)

    print(len(train_graphene_dataloader))
    print(len(val_graphene_dataloader))
    print(len(test_graphene_dataloader))