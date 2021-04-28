import os
import cv2
import json
import config
import numpy as np
from askfile import ask_dir


def read_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def custom_dataset(root_path, save_path):
    for file_name in os.listdir(root_path):
        if file_name.endswith('.jpg'):
            name = file_name.split('.jpg')[0]
            json_name = name + '.json'
            json_path = os.path.join(root_path, json_name)
            image_path = os.path.join(root_path, file_name)
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
                box = [bx0, by0, bx1, by1]
                label_image = np.zeros(shape=(width, height), dtype='uint8')
                for label in labels:
                    lab = label[0]
                    points = label[1]
                    points = np.around(points)
                    points = points.astype(np.int32)
                    # label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab / config.CLASS_NUMBER * 255))
                    label_image[:, :] = cv2.fillPoly(label_image[:, :], pts=[points], color=int(lab))
                cv2.imwrite(os.path.join(save_path, name + '.png'), label_image)
    

if __name__ == '__main__':
    custom_dataset(ask_dir(), ask_dir())