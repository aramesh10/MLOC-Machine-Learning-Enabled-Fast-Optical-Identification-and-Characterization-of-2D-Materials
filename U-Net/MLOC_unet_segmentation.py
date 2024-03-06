import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from dataset import GrapheneDataset, post_process_mask_prediction

RUN_NAME = "dice_iou_preprocess"
EPOCHS = 20
BATCH_SIZE = 4

if __name__ == '__main__':
    # Create directory if not exist
    if not os.path.exists(f"./U-Net/runs/{RUN_NAME}"):
        os.makedirs(f"./U-Net/runs/{RUN_NAME}")
        os.makedirs(f"./U-Net/runs/{RUN_NAME}/figures")
        print(f"Created ./U-Net/runs/{RUN_NAME} directory")

    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'

    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER']
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cpu'

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )

    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir, preprocess_bool=True)
    val_graphene_dataset   = GrapheneDataset(x_val_dir, y_val_dir, preprocess_bool=True)
    test_graphene_dataset  = GrapheneDataset(x_test_dir, y_test_dir, preprocess_bool=True)

    train_graphene_dataloader = DataLoader(train_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_graphene_dataloader = DataLoader(val_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_graphene_dataloader = DataLoader(test_graphene_dataset, batch_size=BATCH_SIZE, shuffle=True)
    max_score = 0

    loss = smp.losses.DiceLoss(mode='multiclass')
    loss.__name__ = 'Dice Loss'
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # Train
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    train_jaccard_loss = []
    train_iou_score = []
    valid_jaccard_loss = []
    valid_iou_score = []
    for i in range(0, EPOCHS):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_graphene_dataloader)
        valid_logs = valid_epoch.run(val_graphene_dataloader)
        
        # Append results
        train_jaccard_loss.append(train_logs[loss.__name__])
        train_iou_score.append(train_logs[metrics[0].__name__])
        valid_jaccard_loss.append(valid_logs[loss.__name__])
        valid_iou_score.append(valid_logs[metrics[0].__name__])
    
        if max_score < valid_logs[metrics[0].__name__]:
            max_score = valid_logs[metrics[0].__name__]
            torch.save(model, f'./U-Net/runs/{RUN_NAME}/best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    torch.save(model, f'./U-Net/runs/{RUN_NAME}/last_model.pth')
    print('Last Model saved!')

    # Plot and save loss and score
    plt.figure()
    plt.title(f"Train - {loss.__name__}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss.__name__}")
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 1.0)
    plt.plot(range(EPOCHS), train_jaccard_loss)
    plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/train_jaccard_loss.png")

    plt.figure()
    plt.title(f"Train - {metrics[0].__name__}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metrics[0].__name__}")
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 1.0)
    plt.plot(range(EPOCHS), train_iou_score)
    plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/train_iou_score.png")

    plt.figure()
    plt.title(f"Valid - {loss.__name__}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss.__name__}")
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 1.0)
    plt.plot(range(EPOCHS), valid_jaccard_loss)
    plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/valid_jaccard_loss.png")

    plt.figure()
    plt.title(f"Valid - {metrics[0].__name__}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metrics[0].__name__}")
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 1.0)
    plt.plot(range(EPOCHS), valid_iou_score)
    plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/valid_iou_score.png")

    plt.figure()
    plt.title("Train & Valid - Loss and Score")
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss.__name__} & {metrics[0].__name__}")
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 1.0)
    plt.plot(range(EPOCHS), train_jaccard_loss, label=f'Train {loss.__name__}')
    plt.plot(range(EPOCHS), train_iou_score, label=f'Train {metrics[0].__name__}')
    plt.plot(range(EPOCHS), valid_jaccard_loss, label=f'Valid {loss.__name__}')
    plt.plot(range(EPOCHS), valid_iou_score, label=f'Valid {metrics[0].__name__}')
    plt.legend()
    plt.savefig(f"./U-Net/runs/{RUN_NAME}/figures/training_results.png")

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