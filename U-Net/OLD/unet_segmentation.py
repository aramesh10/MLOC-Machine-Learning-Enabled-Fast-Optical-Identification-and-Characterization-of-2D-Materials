import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import torchvision.transforms as transforms
from dataset2 import GrapheneDataset#, GrapheneDataloader, visualize
import matplotlib.pyplot as plt

EPOCH = 3
BATCH_SIZE = 2
SHAPE = (128, 128)

if __name__ == '__main__':
    x_train_dir = './data/train/images/'
    y_train_dir = './data/train/labels/'
    
    x_val_dir = './data/val/images/'
    y_val_dir = './data/val/labels/'

    x_test_dir = './data/test/images/'
    y_test_dir = './data/test/labels/'

    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER']
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cpu'

    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    transform = transforms.Compose([transforms.ToTensor()])

    train_graphene_dataset = GrapheneDataset(x_train_dir, y_train_dir)#, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])
    val_graphene_dataset = GrapheneDataset(x_val_dir, y_val_dir)#, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])
    test_graphene_dataset = GrapheneDataset(x_test_dir, y_test_dir)#, classes=['SUBSTRATE', 'MONO-LAYER', 'MULTI-LAYER'])
    
    # train_graphene_dataloader = GrapheneDataloader(train_graphene_dataset, batch_size=1, shuffle=True)
    # val_graphene_dataloader = GrapheneDataloader(val_graphene_dataset, batch_size=1, shuffle=True)
    # test_graphene_dataloader = GrapheneDataloader(test_graphene_dataset, batch_size=1, shuffle=True)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

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

    # Train model
    max_score = 0
    for i in range(0, EPOCH):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_graphene_dataset)
        valid_logs = valid_epoch.run(val_graphene_dataset)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # Test model
    best_model = torch.load('./best_model.pth')
    
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    # logs = test_epoch.run(test_graphene_dataloader)
    # print(logs)

    # for _ in range(5):
    #     n = np.random.choice(len(test_dataloader))

    #     image, label = test_data[n]
    #     x_tensor = image.to(DEVICE).unsqueeze(0)
    #     prediction_mask = best_model.predict(x_tensor)

    #     print('*'*50)
    #     print("Prediction mask:")
    #     print(prediction_mask)
    #     print(f"shape: {prediction_mask.shape}")
    #     print('*'*50)

    #     plt.figure()
    #     plt.imshow(np.array(image).reshape(96, 96, 3))
    #     plt.figure()
    #     plt.imshow(np.array(label).reshape(96, 96, 3))
    #     plt.figure()
    #     plt.imshow(np.array(prediction_mask).reshape(96, 96, 3))
    #     plt.show()

    for i in range(5):
        n = np.random.choice(len(test_graphene_dataset))
        
        # image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_graphene_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        # visualize(
        #     image=image, 
        #     ground_truth_mask=gt_mask, 
        #     predicted_mask=pr_mask
        # )

        plt.figure()
        plt.imshow(np.array(x_tensor).reshape(96, 96, 3))
        plt.figure()
        plt.imshow(np.array(gt_mask).reshape(96, 96, 3))
        plt.figure()
        plt.imshow(np.array(pr_mask).reshape(96, 96, 3))
        plt.show()