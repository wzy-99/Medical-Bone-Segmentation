import os
import cv2
import json
import numpy as np
import paddle.fluid as fluid
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, ColorJitter
import config
from random_crop import RandomCrop


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.1),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), 
    Transpose(),
])


croper = RandomCrop(min_ratio=config.MIN_CROP_RATIO, max_ratio=config.MAX_CROP_RATIO)


def read_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


class TrainDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.sample = []
        for file_name in os.listdir(self.root_path):
            if file_name.endswith('.jpg'):
                name = file_name.split('.jpg')[0]
                json_name = name + '.json'
                json_path = os.path.join(self.root_path, json_name)
                image_path = os.path.join(self.root_path, file_name)
                if os.path.exists(json_path):
                    js = read_json(json_path)
                    shapes = js['shapes']
                    labels = []
                    for shape in shapes:
                        lab = config.LABEL2ID[shape['label']]
                        shape_type = shape['shape_type']
                        if shape_type == 'polygon':
                            points = shape['points']
                            points = np.array(points)
                            labels.append([lab, points])
                    self.sample.append({'image_path': image_path, 'labels': labels})
    
    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        h, w, c = image.shape
        label_image = np.zeros(shape=(config.LABLE_SIZE, config.LABLE_SIZE), dtype='int32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1]
            points[:, 0] = points[:, 0] / w * config.LABLE_SIZE
            points[:, 1] = points[:, 1] / h * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab))
            # cv2.imwrite('gray.jpg', (label_image * 255).astype(np.uint8))
        image = transform(image).astype("float32")
        ret = image, label_image
        return ret
    
    def __len__(self):
        return len(self.sample)


class ValidDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.sample = []
        for file_name in os.listdir(self.root_path):
            if file_name.endswith('.jpg'):
                name = file_name.split('.jpg')[0]
                json_name = name + '.json'
                json_path = os.path.join(self.root_path, json_name)
                image_path = os.path.join(self.root_path, file_name)
                if os.path.exists(json_path):
                    js = read_json(json_path)
                    shapes = js['shapes']
                    labels = []
                    for shape in shapes:
                        lab = config.LABEL2ID[shape['label']]
                        shape_type = shape['shape_type']
                        if shape_type == 'polygon':
                            points = shape['points']
                            points = np.array(points)
                            labels.append([lab, points])
                    self.sample.append({'image_path': image_path, 'labels': labels})
    
    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        h, w, c = image.shape
        label_image = np.zeros(shape=(config.LABLE_SIZE, config.LABLE_SIZE), dtype='int32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1]
            points[:, 0] = points[:, 0] / w * config.LABLE_SIZE
            points[:, 1] = points[:, 1] / h * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab))
            # cv2.imwrite('gray.jpg', (label_image * 255).astype(np.uint8))
        image = transform(image).astype("float32")
        ret = image, label_image
        return ret
    
    def __len__(self):
        return len(self.sample)


class TestDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.sample = []
        self.indexs = []

        for file_name in os.listdir(self.image_path):
            if file_name.endswith('.jpg'):
                self.sample.append(os.path.join(self.image_path, file_name))
        
    def __getitem__(self, idx):
        self.indexs.append(self.sample[idx])
        image = cv2.imread(self.sample[idx])
        image = transform(image).astype("float32")
        ret = image
        return ret

    def __len__(self):
        return len(self.sample)