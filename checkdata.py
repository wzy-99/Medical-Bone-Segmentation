import os
import cv2
import json
import numpy as np
from showbox import ShowBox
from askbox import ask_dir

origin = ShowBox('origin')
result = ShowBox('result')


def read_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def check(file_name, image_path, json_path):
    img = cv2.imread(image_path)
    origin.show(img)
    js = read_json(json_path)
    shapes = js["shapes"]  # shapes - lsit
    for shape in shapes:   # shape - dict
        shape_type = shape['shape_type']
        points = shape['points']  # points - list
        lab = shape['label']
        print(lab)
        points = np.array(points)
        points = np.around(points)  # 四舍五入
        points = points.astype(np.int32)
        if shape_type == 'rectangle':
            x0y0 = tuple(points[0])
            x1y1 = tuple(points[1])
            cv2.rectangle(img, pt1=x0y0, pt2=x1y1, color=(0, 0, 255), thickness=3)
        elif shape_type == 'polygon':
            cv2.fillPoly(img, pts=[points], color=(0, 0, 255))
            cv2.polylines(img, pts=[points], isClosed=True, color=(0, 255, 0), thickness=3)
        else:
            print(shape_type)
    result.show(img)
    cv2.waitKey(0)
    # cv2.imwrite(os.path.join('./result', file_name + '.jpg'), img)


def check_all(path):
    for image_name in os.listdir(path):
        if image_name.endswith('.jpg'):
            file_name = image_name.split('.jpg')[0]
            json_name = file_name + '.json'
            image_path = os.path.join(path, image_name)
            json_path = os.path.join(path, json_name)
            if os.path.exists(json_path):
                check(file_name, image_path, json_path)


if __name__ == '__main__':
    check_all(ask_dir())
