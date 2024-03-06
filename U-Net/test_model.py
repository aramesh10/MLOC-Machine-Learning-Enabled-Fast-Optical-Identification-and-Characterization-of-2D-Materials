import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from dataset import GrapheneDataset, post_process_mask_prediction #, image_to_tensor, label_to_tensor, image_normalize

RUN_NAME = "jaccard_iou_preprocess" # "jaccard_iou_preprocess" "jaccard_iou_no_preprocess"
# EPOCHS = 20
BATCH_SIZE = 4

if __name__ == '__main__':
    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER']
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cpu'

    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=True)
    test_graphene_dataloader = DataLoader(test_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    loss = smp.losses.JaccardLoss(mode='multiclass')
    loss.__name__ = 'Jaccard Loss'
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Test
    best_model = torch.load(f"./U-Net/runs/{RUN_NAME}/best_model.pth")
    last_model = torch.load(f"./U-Net/runs/{RUN_NAME}/last_model.pth")
    models = [best_model, last_model]
    models_label = ["best_model", "last_model"]
    
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    test_logs = test_epoch.run(test_graphene_dataloader)
    print(test_logs)

    for model_num, model in enumerate(models):
        for i in range(5):
            n = np.random.choice(len(test_graphene_dataloader))

            image, label = test_graphene_dataset[n]
            x_tensor = image.to(DEVICE).unsqueeze(0)
            prediction_mask = model.predict(x_tensor)
            prediction_mask = post_process_mask_prediction(prediction_mask)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle('Model Training Results')
            ax1.imshow(np.array(image).reshape(96, 96))
            ax1.set_title("Image")
            ax2.imshow(np.array(label).reshape(96, 96))
            ax2.set_title("True mask")
            ax3.imshow(prediction_mask)
            ax3.set_title("Model's prediction")
            plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/{models_label[model_num]}_example_{i}.png")