import os
import cv2
import json
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, ColorJitter
import config
import random


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.1),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), 
    Transpose(),
])


def random_region(width, height, box):
    x0, y0, x1, y1 = [int(i) for i in box]
    bw, bh = x1 - x0, y1 - y0

    w_min_ratio = max(config.W_MIN_RATIO, bw / width)
    if w_min_ratio > config.W_MAX_RATIO:
        print('MAX_RATIO is too low')
        w_max_ratio = 1.0
    else:
        w_max_ratio = config.W_MAX_RATIO

    h_min_ratio = max(config.H_MIN_RATIO, bh / height)
    if w_min_ratio > config.H_MAX_RATIO:
        print('MAX_RATIO is too low')
        w_max_ratio = 1.0
    else:
        h_max_ratio = config.H_MAX_RATIO

    w_ratio = random.random() * (w_max_ratio - w_min_ratio) + w_min_ratio
    h_ratio = random.random() * (h_max_ratio - h_min_ratio) + h_min_ratio

    w = int(width * w_ratio)
    h = int(height * h_ratio)

    lmin = max(0, x1 - w)
    lmax = min(width - 1 - w, x0)
    tmin = max(0, y1 - h)
    tmax = min(height -1 - h, y0)

    left = random.randint(lmin, lmax)
    top = random.randint(tmin, tmax)
    
    return left, top, w, h


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
                    width = js['imageWidth']
                    height = js['imageHeight']
                    shapes = js['shapes']
                    labels = []
                    for shape in shapes:
                        lab = config.LABEL2ID[shape['label']]
                        shape_type = shape['shape_type']
                        if shape_type == 'polygon':
                            points = shape['points']
                            points = np.array(points)
                            labels.append([lab, points])
                    px = [lab[1][0] for lab in labels]
                    py = [lab[1][1] for lab in labels]
                    px = np.array(px)
                    py = np.array(py)
                    bx0 = px.min()
                    bx1 = px.max()
                    by0 = py.min()
                    by1 = py.max()
                    bx0 = max(0, bx0 - config.MARGIN)
                    by0 = max(0, by0 - config.MARGIN)
                    bx1 = min(width - 1, bx1 + config.MARGIN)
                    by1 = min(height - 1, by1 + config.MARGIN)
                    self.sample.append({'image_path': image_path, 'labels': labels, 'box': [bx0, by0, bx1, by1]})
                        
    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        height, width, channel = image.shape
        nx0, ny0, nw, nh = random_region(width, height, self.sample[idx]['box'])
        nx1 = nx0 + nw
        ny1 = ny0 + nh
        dat = image[ny0:ny1, nx0:nx1, :]
        label_image = np.zeros(shape=(config.LABLE_SIZE, config.LABLE_SIZE), dtype='int32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1].copy()
            points[:, 0] = (points[:, 0] - nx0) / nw * config.LABLE_SIZE
            points[:, 1] = (points[:, 1] - ny0) / nh * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab))
        dat = transform(dat).astype("float32")
        ret = dat, label_image
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
                    width = js['imageWidth']
                    height = js['imageHeight']
                    shapes = js['shapes']
                    labels = []
                    for shape in shapes:
                        lab = config.LABEL2ID[shape['label']]
                        shape_type = shape['shape_type']
                        if shape_type == 'polygon':
                            points = shape['points']
                            points = np.array(points)
                            labels.append([lab, points])
                    px = [lab[1][0] for lab in labels]
                    py = [lab[1][1] for lab in labels]
                    px = np.array(px)
                    py = np.array(py)
                    bx0 = px.min()
                    bx1 = px.max()
                    by0 = py.min()
                    by1 = py.max()
                    bx0 = max(0, bx0 - config.MARGIN)
                    by0 = max(0, by0 - config.MARGIN)
                    bx1 = min(width - 1, bx1 + config.MARGIN)
                    by1 = min(height - 1, by1 + config.MARGIN)
                    self.sample.append({'image_path': image_path, 'labels': labels, 'box': [bx0, by0, bx1, by1]})
                        
    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        height, width, channel = image.shape
        nx0, ny0, nw, nh = random_region(width, height, self.sample[idx]['box'])
        nx1 = nx0 + nw
        ny1 = ny0 + nh
        dat = image[ny0:ny1, nx0:nx1, :]
        label_image = np.zeros(shape=(config.LABLE_SIZE, config.LABLE_SIZE), dtype='int32')
        for label in self.sample[idx]['labels']:
            lab = label[0]
            points = label[1].copy()
            points[:, 0] = (points[:, 0] - nx0) / nw * config.LABLE_SIZE
            points[:, 1] = (points[:, 1] - ny0) / nh * config.LABLE_SIZE
            points = np.around(points)
            points = points.astype(np.int32)
            label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab))
        dat = transform(dat).astype("float32")
        ret = dat, label_image
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
                self.sample.append({'image_path': os.path.join(self.image_path, file_name)})
        
    def __getitem__(self, idx):
        self.indexs.append(self.sample[idx])
        image = cv2.imread(self.sample[idx]['image_path'])
        image = transform(image).astype("float32")
        ret = image
        return ret

    def __len__(self):
        return len(self.sample)


class TestDataset2(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.sample = []
        self.indexs = []
        for file_name in os.listdir(self.root_path):
            if file_name.endswith('.jpg'):
                name = file_name.split('.jpg')[0]
                json_name = name + '.json'
                json_path = os.path.join(self.root_path, json_name)
                image_path = os.path.join(self.root_path, file_name)
                if os.path.exists(json_path):
                    js = read_json(json_path)
                    width = js['imageWidth']
                    height = js['imageHeight']
                    shapes = js['shapes']
                    labels = []
                    for shape in shapes:
                        lab = config.LABEL2ID[shape['label']]
                        shape_type = shape['shape_type']
                        if shape_type == 'polygon':
                            points = shape['points']
                            points = np.array(points)
                            labels.append([lab, points])
                    px = [lab[1][0] for lab in labels]
                    py = [lab[1][1] for lab in labels]
                    px = np.array(px)
                    py = np.array(py)
                    bx0 = px.min()
                    bx1 = px.max()
                    by0 = py.min()
                    by1 = py.max()
                    bx0 = max(0, bx0 - config.MARGIN)
                    by0 = max(0, by0 - config.MARGIN)
                    bx1 = min(width - 1, bx1 + config.MARGIN)
                    by1 = min(height - 1, by1 + config.MARGIN)
                    self.sample.append({'image_path': image_path, 'labels': labels, 'box': [bx0, by0, bx1, by1]})
                        
    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        height, width, channel = image.shape
        nx0, ny0, nw, nh = random_region(width, height, self.sample[idx]['box'])
        nx1 = nx0 + nw
        ny1 = ny0 + nh
        dat = image[ny0:ny1, nx0:nx1, :]
        dat = transform(dat).astype("float32")
        ind = self.sample[idx].copy()
        ind['region'] = [nx0, ny0, nx1, ny1]
        self.indexs.append(ind)
        ret = dat
        return ret
    
    def __len__(self):
        return len(self.sample)


if __name__ == '__main__':
    paddle.seed(1)
    random.seed(1)
    ds = TrainDataset('train')
    cv2.namedWindow('o', cv2.WINDOW_NORMAL)
    while True:
        for i, dat in enumerate(ds):
            x, label = dat
            print(label.sum())
            print(ds.sample[i]['image_path'])
            img = cv2.imread(ds.sample[i]['image_path'].replace('train', 'result_train'))
            cv2.imshow('o', img)
            cv2.imshow('x', (x[0] + 0.5))
            cv2.imshow('l', (label / config.CLASS_NUMBER * 255).astype('uint8'))
            cv2.waitKey(0)