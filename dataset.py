import os
import cv2
import json
import numpy as np
import paddle.fluid as fluid
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
import config


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), # 标准化
    Transpose(), # 原始数据形状维度是HWC格式，经过Transpose，转换为CHW格式
])


def read_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


class TrainDataset(Dataset):
    def __init__(self, root_path, class_number=1):
        self.root_path = root_path
        self.class_number = class_number
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
        label_image = np.zeros(shape=(self.class_number, config.LABLE_SIZE, config.LABLE_SIZE), dtype='float32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1]
            points[:, 0] = points[:, 0] / w * config.LABLE_SIZE
            points[:, 1] = points[:, 1] / h * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[c, :, :] = cv2.fillPoly(label_image[c, :, :], pts=[points], color=1.0)
            # cv2.imwrite('gray.jpg', (label_image[c] * 255).astype(np.uint8))
        image = transform(image).astype("float32")
        ret = image, label_image
        return ret
    
    def __len__(self):
        return len(self.sample)


class ValidDataset(Dataset):
    def __init__(self, root_path, class_number=1):
        self.root_path = root_path
        self.class_number = class_number
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
        label_image = np.zeros(shape=(self.class_number, config.LABLE_SIZE, config.LABLE_SIZE), dtype='float32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1]
            points[:, 0] = points[:, 0] / w * config.LABLE_SIZE
            points[:, 1] = points[:, 1] / h * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[c, :, :] = cv2.fillPoly(label_image[c, :, :], pts=[points], color=1.0)
            # cv2.imwrite('gray.jpg', (label_image[c] * 255).astype(np.uint8))
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