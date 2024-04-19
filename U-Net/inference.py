import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from dataset import GrapheneDataset, post_process_mask_prediction

RUN_NAME = "dice_iou_hsv_normalized_preprocess"
BATCH_SIZE = 4
DEVICE = 'cpu'

if __name__ == '__main__':
    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=True, img_format='hsv', normalize=True)
    test_graphene_dataloader = DataLoader(test_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_model = torch.load(f"./U-Net/runs/{RUN_NAME}/best_model.pth")

    for i in range(100):
        n = np.random.choice(len(test_graphene_dataloader))

        image, label = test_graphene_dataset[n]
        x_tensor = image.to(DEVICE).unsqueeze(0)

        prediction_mask = best_model.predict(x_tensor)
        prediction_mask = post_process_mask_prediction(prediction_mask)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Model Training Results')
        ax1.imshow(np.array(image).reshape(96, 96))
        ax1.set_title("Original Image")
        ax2.imshow(np.array(label).reshape(96, 96))
        ax2.set_title("Labels")
        ax3.imshow(prediction_mask)
        ax3.set_title("Prediction")
        plt.show()