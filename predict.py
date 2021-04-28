import os
import cv2
import paddle
import numpy as np
import random
import config
from dataset import TestDataset, TestDataset2
from model import Unet


def predict():
    # testdataset = TestDataset('test')
    testdataset = TestDataset2('train')
    net = Unet(config.CLASS_NUMBER)
    model = paddle.Model(net)
    model.load('output/unet1')
    model.prepare()
    outs = model.predict(testdataset)
    outs = outs[0]
    for index, out in enumerate(outs):
        path = testdataset.indexs[index]['image_path']
        x0, y0, x1, y1 = testdataset.indexs[index]['region']
        img = cv2.imread(path)
        img = img[y0:y1, x0:x1]
        
        # 输出热力图
        res = np.reshape(out, (config.LABLE_SIZE, config.LABLE_SIZE, config.CLASS_NUMBER))
        lab = (np.argmax(res, axis=-1) / config.CLASS_NUMBER * 255).astype(np.uint8)
        color = cv2.applyColorMap(lab, cv2.COLORMAP_JET)
        cv2.imwrite('result/' + str(index) + 'result.jpg', color)

        # 输出每张图
        res = np.reshape(out, (config.LABLE_SIZE, config.LABLE_SIZE, config.CLASS_NUMBER))
        res = res * 254
        res = res.astype(np.uint8)
        for ch in range(config.CLASS_NUMBER):
            r = res[:, :, ch]
            lab = config.ID2LABEL[ch]
            cv2.imwrite('result/' + str(index) + 'result' + lab + '.jpg', r)
        
        cv2.imwrite('result/' + str(index) + 'origin.jpg', img)


if __name__ == '__main__':
    paddle.seed(1)
    random.seed(1)
    predict()
