from ultralytics import YOLO
import skimage
import pandas as pd
import matplotlib.image as img

if __name__ == '__main__':
    label_image_dir = '../data/train/images/7_data0001_normalized.jpg'    
    label_image = img.imread(label_image_dir)

    props = skimage.measure.regionprops_table(label_image, properties=['label', 'centroid'])
    props_dataframe = pd.DataFrame(props)

    print(props_dataframe)

    # model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    # results = model.train(data='', epochs=100, imgsz=640)