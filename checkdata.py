import os
import cv2
import json
import numpy as np
from showbox import ShowBox
from askbox import ask_dir

origin = ShowBox('origin')
result = ShowBox('result')

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

widths = []
heights = []
labels = []

def read_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def check(file_name, image_path, json_path):
    img = cv2.imread(image_path)
    origin.show(img)
    js = read_json(json_path)
    shapes = js["shapes"]  # shapes - lsit
    for index, shape in enumerate(shapes):   # shape - dict
        shape_type = shape['shape_type']
        points = shape['points']  # points - list
        lab = shape['label']
        labels.append(lab) if lab not in labels else None
        points = np.array(points)
        points = np.around(points)  # 四舍五入
        points = points.astype(np.int32)
        if shape_type == 'rectangle':
            widths.append(points[1][0] - points[0][0])
            heights.append(points[1][1] - points[0][1])
            x0y0 = tuple(points[0])
            x1y1 = tuple(points[1])
            cv2.rectangle(img, pt1=x0y0, pt2=x1y1, color=colors[index % 3], thickness=3)
        elif shape_type == 'polygon':
            cv2.fillPoly(img, pts=[points], color=[(1 + index) % 3])
            cv2.polylines(img, pts=[points], isClosed=True, color=colors[index % 3], thickness=3)
    result.show(img)
    key = cv2.waitKey(0)
    if key == ord('q') or key == ord('Q'):
        return False
    # cv2.imwrite(os.path.join('./result', file_name + '.jpg'), img)


def check_all(path):
    for image_name in os.listdir(path):
        if image_name.endswith('.jpg'):
            file_name = image_name.split('.jpg')[0]
            json_name = file_name + '.json'
            image_path = os.path.join(path, image_name)
            json_path = os.path.join(path, json_name)
            if os.path.exists(json_path):
                if check(file_name, image_path, json_path) is False:
                    break


if __name__ == '__main__':
    check_all(ask_dir())
    print(widths)
    print(heights)
    widths = np.array(widths)
    heights = np.array(heights)
    print(widths.max(), widths.min()) # 1024
    print(heights.max(), heights.min()) # 1024
    print(labels)